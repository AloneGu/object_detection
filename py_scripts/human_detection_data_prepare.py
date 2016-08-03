import os
import sys
import numpy as np
import cv2
from time import sleep
from random import randint

# resized image size
TRAINING_W = 96
TRAINING_H = 96
NEG_IMAGE_INTERVAL = 20
REGION_PIX = 15


def get_place_id(video):
    """
        Extracts camera ID from video filenames.
    """
    import re

    m = re.search('\d\d\d\d\d\d\d', video)
    if m:
        return m.group()
    else:
        return None


def process_sizes(sizes_file):
    """
        Parses a sizes file and returns a dictionary of bin to size.
    """
    try:
        guide_file = open(sizes_file, 'r')
    except IOError:
        print "No valid sizes file found"
        sys.exit()
    sizes = {}
    for line in guide_file:
        line = line.split()
        sizes[(int(line[0]), int(line[1]))] = int(line[2])
    return sizes


class human_detection_prepare:
    def __init__(self, place_id):
        self.place_id = place_id
        self.work_pos_dir = '../training_set/' + self.place_id + '/pos/'
        self.work_neg_dir = '../training_set/' + self.place_id + '/neg/'
        if not os.path.exists(self.work_pos_dir):
            os.makedirs(self.work_pos_dir)
        if not os.path.exists(self.work_neg_dir):
            os.makedirs(self.work_neg_dir)

    def consume_pos_img(self, roi, frame_number, video_fn):
        file_name = os.path.join(self.work_pos_dir, video_fn + '_' + str(frame_number) + '.png')
        print 'save one pos img', file_name
        roi = cv2.resize(roi, (TRAINING_W, TRAINING_H))
        cv2.imwrite(file_name, roi)

    def consume_neg_img(self, roi, frame_number, video_fn):
        file_name = os.path.join(self.work_neg_dir, video_fn + '_' + str(frame_number) + '.png')
        print 'save one neg img', file_name
        # roi=cv2.resize(roi,(TRAINING_W,TRAINING_H))
        cv2.imwrite(file_name, roi)


def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


def training_img_process(video, ROTATE):
    """
    Performs zone counting and in-out counting on VIDEO.  LOI_BOX is a 4-tuple
    consisting of the x-and y-coordinates of the upper-left corner of the LoI box,
    along with its width and height.  VIEW_BOX is 4-tuple consiting of the x- and
    y-coordinates of the zone of interest, along with its width and height.  INOUT
    is a 2-tuple of booleans where the first is whether in-out traffic is up/down
    (True) or left/right (False) and the second is whether in is up/right (True) or
    down/left (False).
    """

    WIDTH_MIN, WIDTH_MAX, HEIGHT_MIN, HEIGHT_MAX = (0, 480, 0, 320)

    sizes = process_sizes('../binsize_files/sizes_output_' + get_place_id(video) + '.txt')
    video_file_name = os.path.splitext(os.path.split(video)[1])[0]
    image_collector = human_detection_prepare(get_place_id(video))
    cap = cv2.VideoCapture(video)

    # Wei added for checking size.txt index range
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
    last_count = 10
    _, frame = cap.read()
    cols, rows, _ = frame.shape
    max_len = int(np.sqrt(cols * cols + rows * rows) + 1)
    # M is the matrix controls the rotation, center is the image's center, (max_len/2,max_len/2)
    M = cv2.getRotationMatrix2D((max_len / 2, max_len / 2), ROTATE, 1)
    # VideoCapture::get: returns the specified VideoCapture property
    # CV_CAP_PROP_FRAME_COUNT: Number of frames in the video file
    # video_length is the total frames' number
    video_length = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    while (count < video_length - 10):
        count += 1
        ret, frame = cap.read()
        # use org_frame to generate neg img
        org_frame = frame
        # process after 50 frames
        if count > 50:
            if ROTATE != 0:
                frame = cv2.copyMakeBorder(frame, (max_len - cols) / 2, (max_len - cols) / 2, \
                                           (max_len - rows) / 2, (max_len - rows) / 2, cv2.BORDER_CONSTANT, value=0)
                frame = cv2.warpAffine(frame, M, (max_len, max_len))
            frame = frame[HEIGHT_MIN:HEIGHT_MAX, WIDTH_MIN:WIDTH_MAX, :]
            # use org_view_box_frame to generate pos img
            org_view_box_frame = frame
            frame = cv2.blur(frame, (3, 3))
            fgmask = fgbg.apply(frame, learningRate=0.001)

            if count > 1600:
                if count % 10 == 0:
                    continue
                max_area, people_count = 0, 0
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
                    tmp_size = 800 * DOUBLE_RATIO
                    if centroid in sizes:
                        tmp_size = sizes[centroid] * DOUBLE_RATIO
                    if area_p > tmp_size:
                        people_count += 2
                        bounding_boxes_size_dict[(x, y, w, h)] = 2
                        good_contour_areas.append(area_p)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        bounding_boxes_dict[(x, y, w, h)] = (NO_PENALTY, NO_PENALTY)

                        human_possibility = 0
                        # print 'possible box',w,h
                        if min(w, h) / (max(w, h) * 1.0) > 0.6:
                            human_possibility = 1
                        if human_possibility == 1:
                            if y < REGION_PIX or x < REGION_PIX or y + h + REGION_PIX > video_height or x + w + REGION_PIX > video_width:
                                pass
                            else:
                                if count - last_count < 10:
                                    continue
                                else:
                                    last_count = count
                                roi = org_view_box_frame[y - REGION_PIX:y + h + REGION_PIX,
                                      x - REGION_PIX:x + w + REGION_PIX]
                                # print roi.shape
                                image_collector.consume_pos_img(roi, count, video_file_name)

                    elif area_p > 200:
                        people_count += 1
                        bounding_boxes_size_dict[(x, y, w, h)] = 1
                        good_contour_areas.append(area_p)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        bounding_boxes_dict[(x, y, w, h)] = (NO_PENALTY, NO_PENALTY)

                if count % NEG_IMAGE_INTERVAL == 1:
                    print count, len(contours), max_area, len(good_contour_areas), good_contour_areas
                    # generate neg image
                    if len(contours) <= 5:
                        # the video is 480*320, so x+width<480 and y+height<320
                        tmp_x = randint(0, 350)
                        tmp_y = randint(0, 200)
                        roi = org_frame[tmp_y:tmp_y + 96, tmp_x:tmp_x + 96]
                        image_collector.consume_neg_img(roi, count, video_file_name)

                draw_str(frame, (20, 20), str(people_count))

                cv2.imshow('orig', frame)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
            else:
                if count % 400 == 0:
                    print count

    cap.release()
    cv2.destroyAllWindows()
    sleep(20)


if len(sys.argv) != 2:
    print "Usage: python human_detection_data_prepare.py VIDEO"
    print "Or"
    print "Usage: python human_detection_data_prepare.py video folder (./videos/download_videos_8600154/)"

DIVISION_RATIO = 40
TRACK_LEN = 10
DOUBLE_RATIO = 1.8
DETECT_INTERVAL = 5
NO_PENALTY = 0
WEAK_PENALTY = 1
STRONG_PENALTY = 2

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

ROTATE = False
if sys.argv[1].endswith('.3gp') or sys.argv[1].endswith('.mp4'):
    training_img_process(sys.argv[1], ROTATE)
else:
    videos_files = os.listdir(sys.argv[1])
    for each_video in videos_files:
        if each_video.endswith('.3gp') or each_video.endswith('.mp4'):
            print 'processing', each_video
            training_img_process(sys.argv[1] + each_video, ROTATE)
