import os
import sys
import numpy as np
import cv2
from time import sleep
from random import randint
import xml.etree.ElementTree as ET
import re
from utilities import get_place_id, process_sizes

WINSIZE = (96, 96)
BLOCKSIZE = (16, 16)
BLOCKSTRIDE = (8, 8)
CELLSIZE = (8, 8)
NBINS = 9
REGION_PIX = 0
TRAINING_W = 96
TRAINING_H = 96
NEG_IMAGE_INTERVAL = 50
PREDICT_AREA_P = 100
XML = '../xmls/stable_8600154.xml'
PREDICT_FLAG = 1
NON_PREDICT_FLAG = 1


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

    def my_predict(self, img):
        img = cv2.resize(img, (96, 96))
        hog_feature = self.hog.compute(img)
        result = self.my_svm.predict(hog_feature)
        img = my_rotate(img, 90)
        hog_feature = self.hog.compute(img)
        result_2 = self.my_svm.predict(hog_feature)
        # print 'predict result',result
        return result or result_2


def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


def zone_counter(video, ROTATE=0):
    """
    Performs zone counting and in-out counting on VIDEO.  LOI_BOX is a 4-tuple
    consisting of the x-and y-coordinates of the upper-left corner of the LoI box,
    along with its width and height.  VIEW_BOX is 4-tuple consiting of the x- and
    y-coordinates of the zone of interest, along with its width and height.  INOUT
    is a 2-tuple of booleans where the first is whether in-out traffic is up/down
    (True) or left/right (False) and the second is whether in is up/right (True) or
    down/left (False).
    """

    VIEW_BOX = (0, 480, 0, 320)
    WIDTH_MIN, WIDTH_MAX, HEIGHT_MIN, HEIGHT_MAX = VIEW_BOX
    print VIEW_BOX
    sizes = process_sizes('../binsize_files/sizes_output_' + get_place_id(video) + '.txt')
    video_file_name = os.path.splitext(os.path.split(video)[1])[0]
    cap = cv2.VideoCapture(video)
    svm_detector = my_detector()
    # Wei added for checking size.txt index range
    speed_up_flag = 0
    video_width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    video_x_index_range = int(video_width / DIVISION_RATIO)
    video_y_index_range = int(video_height / DIVISION_RATIO)
    print 'video width', video_width, 'video height', video_height, 'x range', video_x_index_range, 'y range', video_y_index_range

    # fgbg is a background subtractor
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

        #jump 50 frames
        if speed_up_flag > 0:
            speed_up_flag -= 1
            continue

        #use org_frame to generate neg img
        org_frame = frame
        #process after 50 frames
        if count > 50:
            if ROTATE != 0:
                frame = cv2.copyMakeBorder(frame, (max_len - cols) / 2, (max_len - cols) / 2, \
                                           (max_len - rows) / 2, (max_len - rows) / 2, cv2.BORDER_CONSTANT, value=0)
                frame = cv2.warpAffine(frame, M, (max_len, max_len))
            frame = frame[HEIGHT_MIN:HEIGHT_MAX, WIDTH_MIN:WIDTH_MAX, :]
            #use org_view_box_frame to generate pos img
            org_view_box_frame = frame
            frame = cv2.blur(frame, (3, 3))
            fgmask = fgbg.apply(frame, learningRate=0.001)

            if count > 1600:
                if count % 10 == 0:
                    continue
                max_area, predict_people_count, people_count = 0, 0, 0
                good_contour_areas = []
                bounding_boxes_dict, bounding_boxes_size_dict = {}, {}
                fgmask = cv2.blur(fgmask, (3, 3))
                contoured = np.copy(fgmask)
                contours, _ = cv2.findContours(contoured, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    centroid = ((x + w / 2) / DIVISION_RATIO, (y + h / 2) / DIVISION_RATIO)
                    area_p = 0
                    if area_p > max_area:
                        max_area = area_p
                    for i in range(y, y + h):
                        for j in range(x, x + w):
                            if fgmask[i, j] > 0:
                                area_p += 1
                    if PREDICT_FLAG:
                        bounding_boxes_size_dict[(x, y, w, h)] = 2
                        human_possibility = 0
                        if min(w, h) / (max(w, h) * 1.0) > 0.6 and w * h > 1600:
                            human_possibility = 1
                        if human_possibility == 1:
                            if y < REGION_PIX or x < REGION_PIX or y + h + REGION_PIX > video_height or x + w + REGION_PIX > video_width:
                                pass
                            else:
                                detect_img = org_view_box_frame[y - REGION_PIX:y + h + REGION_PIX,
                                             x - REGION_PIX:x + w + REGION_PIX]
                                if svm_detector.my_predict(detect_img) > 0:
                                    predict_people_count += 1
                                    cv2.rectangle(org_view_box_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    if NON_PREDICT_FLAG:
                        tmp_size = 800 * DOUBLE_RATIO
                        if centroid in sizes:
                            tmp_size = sizes[centroid] * DOUBLE_RATIO
                        if area_p > tmp_size:
                            people_count += 2
                            bounding_boxes_size_dict[(x, y, w, h)] = 2
                            good_contour_areas.append(area_p)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            bounding_boxes_dict[(x, y, w, h)] = (NO_PENALTY, NO_PENALTY)
                        else:
                            if area_p > 400:
                                people_count += 1
                                bounding_boxes_size_dict[(x, y, w, h)] = 1
                                good_contour_areas.append(area_p)
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                bounding_boxes_dict[(x, y, w, h)] = (NO_PENALTY, NO_PENALTY)

                draw_str(frame, (20, 20), str(people_count))
                draw_str(org_view_box_frame, (20, 20), str(predict_people_count))
                cv2.imshow('predict', org_view_box_frame)
                cv2.imshow('non_predict', frame)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
                if k == 32:
                    speed_up_flag = 50
            else:
                if count % 400 == 0:
                    print count

    cap.release()
    cv2.destroyAllWindows()
    sleep(20)


if len(sys.argv) != 2:
    print "Usage: python zone_count_human_detection_test.py VIDEO"
    print "Or"
    print "Usage: python zone_count_human_detection_test.py video folder (./videos/)"

DIVISION_RATIO = 40
TRACK_LEN = 10
DOUBLE_RATIO = 1.8
DETECT_INTERVAL = 5
NO_PENALTY = 0
WEAK_PENALTY = 1
STRONG_PENALTY = 2

if sys.argv[1].endswith('.3gp') or sys.argv[1].endswith('.mp4'):
    zone_counter(sys.argv[1])
else:
    videos_files = os.listdir(sys.argv[1])
    for each_video in videos_files:
        if each_video.endswith('.3gp') or each_video.endswith('.mp4'):
            print 'processing', each_video
            zone_counter(sys.argv[1] + each_video)