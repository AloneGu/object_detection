# this script will save the sample mturk result to proper folder under
# mturk/mturk_images and save the algo result to mturk/algo_result_csv

import os
import sys
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re, datetime
import csv

WINSIZE = (96, 96)
BLOCKSIZE = (16, 16)
BLOCKSTRIDE = (8, 8)
CELLSIZE = (8, 8)
NBINS = 9
REGION_PIX = 0
TRAINING_W = 96
TRAINING_H = 96
PREDICT_AREA_P = 100
XML = '../xmls/stable_8600154.xml'
NON_PREDICT_FLAG = 1

from utilities import get_place_id, process_sizes

MTURK_IMAGE_DIR = '../mturk/mturk_images/'
MTURK_CSV_DIR = '../mturk/algo_result_csv/'

if not os.path.exists(MTURK_CSV_DIR):
    os.makedirs(MTURK_CSV_DIR)

if not os.path.exists(MTURK_IMAGE_DIR):
    os.makedirs(MTURK_CSV_DIR)


def my_rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    r = cv2.warpAffine(img, M, (w, h))
    return r


class my_detector:
    def __init__(self):
        self.my_svm = cv2.SVM()
        self.my_svm.load(XML)
        self.hog = cv2.HOGDescriptor(WINSIZE, BLOCKSIZE, BLOCKSTRIDE, CELLSIZE, NBINS)

        self.multi_hog = cv2.HOGDescriptor(WINSIZE, BLOCKSIZE, BLOCKSTRIDE, CELLSIZE, NBINS)

        tree = ET.parse(XML)
        root = tree.getroot()
        SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0]
        rho = float(root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text)
        svmvec = [[-float(x)] for x in re.sub('\s+', ' ', SVs.text).strip().split(' ')]
        svmvec.append([rho])
        svmvec = np.array(svmvec)

        self.multi_hog.setSVMDetector(svmvec)

    def my_predict(self, img):
        img = cv2.resize(img, (96, 96))
        hog_feature = self.hog.compute(img)
        result = self.my_svm.predict(hog_feature)
        img = my_rotate(img, 90)
        hog_feature = self.hog.compute(img)
        result_2 = self.my_svm.predict(hog_feature)
        #print 'predict result',result
        return result or result_2

    def my_detect_multiscale(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return len(
            self.multi_hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.01, finalThreshold=2)[0])


def get_time(video_filename):
    video_filename = os.path.split(video_filename)[1]
    m = re.search('\d\d\d\d-\d\d-\d\d-\d\d-\d\d-\d\d', video_filename)
    tmp_index = video_filename.find(str(m.group()))
    base_fn = video_filename[:tmp_index]
    base_time = m.group().split('-')
    base_time = [int(x) for x in base_time]
    base_datetime = datetime.datetime(base_time[0], base_time[1], base_time[2], base_time[3], base_time[4],
                                      base_time[5])
    return base_datetime, base_fn


def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


def zone_counter(video, ROTATE=0):
    VIEW_BOX = (0, 480, 0, 320)
    WIDTH_MIN, WIDTH_MAX, HEIGHT_MIN, HEIGHT_MAX = VIEW_BOX
    print VIEW_BOX
    video_file_name = os.path.splitext(os.path.split(video)[1])[0]
    cap = cv2.VideoCapture(video)
    svm_detector = my_detector()
    video_width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    video_x_index_range = int(video_width / DIVISION_RATIO)
    video_y_index_range = int(video_height / DIVISION_RATIO)
    print 'video width', video_width, 'video height', video_height, 'x range', video_x_index_range, 'y range', video_y_index_range

    start_time, base_filename = get_time(video)
    mturk_minute_dict = {}
    mturk_count_dict = {}
    sizes = process_sizes('../binsize_files/sizes_output_' + get_place_id(video) + '.txt')
    #fgbg is a background subtractor
    fgbg = cv2.BackgroundSubtractorMOG(history=200, nmixtures=10, backgroundRatio=0.0001)
    tracks = []
    cross_count_in, cross_count_out = 0, 0
    prev_gray = None
    count = 0
    _, frame = cap.read()
    cols, rows, _ = frame.shape
    max_len = int(np.sqrt(cols * cols + rows * rows) + 1)
    #M is the matrix controls the rotation, center is the image's center, (max_len/2,max_len/2)
    M = cv2.getRotationMatrix2D((max_len / 2, max_len / 2), ROTATE, 1)
    # VideoCapture::get: returns the specified VideoCapture property
    # CV_CAP_PROP_FRAME_COUNT: Number of frames in the video file
    # video_length is the total frames' number
    video_length = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

    while (count < video_length - 10):
        count += 1
        ret, frame = cap.read()
        org_view_box_frame = frame
        mturk_img = np.copy(frame)
        #process after 50 frames
        if count > 50:
            if ROTATE != 0:
                frame = cv2.copyMakeBorder(frame, (max_len - cols) / 2, (max_len - cols) / 2, (max_len - rows) / 2,
                                           (max_len - rows) / 2, cv2.BORDER_CONSTANT, value=0)
                frame = cv2.warpAffine(frame, M, (max_len, max_len))
            frame = frame[HEIGHT_MIN:HEIGHT_MAX, WIDTH_MIN:WIDTH_MAX, :]
            frame = cv2.blur(frame, (3, 3))
            fgmask = fgbg.apply(frame, learningRate=0.001)

            if count > 1600:
                predict_people_count = 0
                people_count = 0
                max_area = 0
                fgmask = cv2.blur(fgmask, (3, 3))
                contoured = np.copy(fgmask)
                contours, _ = cv2.findContours(contoured, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    centroid = ((x + w / 2) / DIVISION_RATIO, (y + h / 2) / DIVISION_RATIO)
                    human_possibility = 0
                    area_p = 0
                    if area_p > max_area:
                        max_area = area_p
                    for i in range(y, y + h):
                        for j in range(x, x + w):
                            if fgmask[i, j] > 0:
                                area_p += 1

                    if min(w, h) / (max(w, h) * 1.0) > 0.6 and w * h > 1600:
                        human_possibility = 1
                    if human_possibility == 1:
                        if y < REGION_PIX or x < REGION_PIX or y + h + REGION_PIX > video_height or x + w + REGION_PIX > video_width:
                            pass
                        else:
                            detect_img = org_view_box_frame[y - REGION_PIX:y + h + REGION_PIX,
                                         x - REGION_PIX:x + w + REGION_PIX]
                            if svm_detector.my_predict(detect_img) > 0:
                                tmp_addition = 0
                                if w > 96 and h > 96:
                                    tmp_addition = svm_detector.my_detect_multiscale(detect_img)
                                if tmp_addition > 1:
                                    predict_people_count += tmp_addition
                                else:
                                    predict_people_count += 1
                                cv2.rectangle(org_view_box_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    if NON_PREDICT_FLAG:

                        tmp_size = 800 * DOUBLE_RATIO
                        if centroid in sizes:
                            tmp_size = sizes[centroid] * DOUBLE_RATIO
                        if area_p > tmp_size:
                            people_count += 2
                        else:
                            if area_p > 400:
                                people_count += 1

                if predict_people_count >= 1 and count % 15 == 0:
                    # 1 minute 3 images
                    diff_second = datetime.timedelta(seconds=count / 15)
                    curr_time = start_time + diff_second
                    curr_minute = curr_time.minute
                    img_count = 1
                    if curr_minute not in mturk_minute_dict:
                        mturk_minute_dict[curr_minute] = 1
                    else:
                        if mturk_minute_dict[curr_minute] == 3:
                            continue
                        else:
                            mturk_minute_dict[curr_minute] += 1
                            img_count = mturk_minute_dict[curr_minute]
                    time_str = str(curr_time).replace(' ', '-').replace(':', '-')
                    curr_img_name = base_filename + time_str + '_000' + str(img_count) + '.jpg'
                    curr_img_name = curr_img_name.replace('video', 'es')
                    print curr_img_name, predict_people_count, people_count
                    cv2.imwrite(MTURK_IMAGE_DIR + curr_img_name, mturk_img)
                    mturk_count_dict[curr_img_name] = [predict_people_count, people_count]

                draw_str(org_view_box_frame, (20, 20), str(predict_people_count) + '  ' + str(people_count))
                cv2.imshow('predict', org_view_box_frame)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
            else:
                if count % 400 == 0:
                    print count
    time_str = str(start_time).replace(' ', '-').replace(':', '-')
    csv_fn = base_filename + time_str + '.csv'
    csv_file = file(MTURK_CSV_DIR + csv_fn, 'w')
    csv_w = csv.writer(csv_file)
    for img_name in mturk_count_dict:
        csv_w.writerow([img_name] + mturk_count_dict[img_name])
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()


DIVISION_RATIO = 40
TRACK_LEN = 10
DOUBLE_RATIO = 1.8
DETECT_INTERVAL = 5
NO_PENALTY = 0
WEAK_PENALTY = 1
STRONG_PENALTY = 2

# usage: python mturk_test.py VIDEO_FILE  or python mturk_test.py VIDEO_FILE_FOLDER

if sys.argv[1].endswith('.3gp') or sys.argv[1].endswith('.mp4'):
    zone_counter(sys.argv[1])
else:
    videos_files = os.listdir(sys.argv[1])
    for each_video in videos_files:
        if each_video.endswith('.3gp') or each_video.endswith('.mp4'):
            print 'processing', each_video
            zone_counter(sys.argv[1] + each_video)
