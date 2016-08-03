import os
import glob
from random import randint
import numpy as np
import cv2

from hog_svm_cls import HogSvmProcessor

VIDEO_DIR = '../videos/8100224/' # where is the video
OUTPUT_DIR = '/tmp/' # where to save result
PLACEMENT = '8100224' # placement name
PROCESS_DATE = '2016-03-05' # date
PERSON_SIZE_MIN = 2400 # min person area
PERSON_SIZE_MAX = 9600 # max person area
ROI_X_MIN, ROX_X_MAX, ROI_Y_MIN, ROI_Y_MAX = 200, 450, 80, 300 # ROI
NEG_IMG_WIDTH = 64 # width and length of negative training data


class TrainingDataGenerator(object):
    def __init__(self, placement, date_time, output_dir, video_dir):
        self.placement = placement
        self.date_time = date_time

        # prepare work dir
        self.work_dir = os.path.join(output_dir, self.placement + '_' + self.date_time)
        self.work_img_dir = os.path.join(self.work_dir, 'imgs')
        self.check_img_dir = os.path.join(self.work_dir, 'check_imgs')
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        if not os.path.exists(self.work_img_dir):
            os.makedirs(self.work_img_dir)
        if not os.path.exists(self.check_img_dir):
            os.makedirs(self.check_img_dir)

        self.old_pos_dict = {}
        self.old_neg_dict = {}
        self.new_pos_dict = {}
        self.new_neg_dict = {}

        self.video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
        print self.video_files

    def run(self):
        for v in self.video_files:
            self.process_one_video(v) # process video, generate imgs and original boxes

        svm_cls = HogSvmProcessor(self.work_img_dir,self.old_pos_dict,self.old_neg_dict) # train a svm model to remove wrong training data
        svm_cls.run()
        self.new_pos_dict,self.new_neg_dict = svm_cls.return_new_res()

        self.save_org_result() # save result

    def process_one_video(self, v):
        fn = os.path.basename(v)
        fn = os.path.splitext(fn)[0]  # get name without suffix
        print fn

        # start to process video
        cap = cv2.VideoCapture(v)
        fgbg = cv2.BackgroundSubtractorMOG()
        count = 0
        video_length = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        while (count < video_length - 10):
            count += 1
            ret, frame = cap.read()
            # use org_frame to generate neg img
            check_frame = frame
            org_frame = frame
            # process after 50 frames
            if count > 50:
                frame = cv2.blur(frame, (3, 3))
                fgmask = fgbg.apply(frame, learningRate=0.001)
                if count > 600:
                    if count % 10 != 0:  # jump every 10 frames
                        continue
                    fgmask = cv2.blur(fgmask, (3, 3))
                    contoured = np.copy(fgmask)
                    contours, _ = cv2.findContours(contoured, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    tmp_img_name = fn + '_' + str(count) + '.jpg'
                    tmp_save_flag = False
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if x < ROI_X_MIN or x > ROX_X_MAX or y < ROI_Y_MIN or y > ROI_Y_MAX:
                            continue
                        tmp_size = w * h
                        if tmp_size < PERSON_SIZE_MAX and tmp_size > PERSON_SIZE_MIN:
                            if tmp_img_name not in self.old_pos_dict:
                                self.old_pos_dict[tmp_img_name] = [[x, y, w, h]]
                            else:
                                self.old_pos_dict[tmp_img_name].append([x, y, w, h])
                            # mark box
                            cv2.rectangle(check_frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
                            print tmp_size, count, fn
                            tmp_save_flag = True
                    if tmp_save_flag:
                        # get one neg box
                        tmp_x = randint(0, 350)
                        tmp_y = randint(0, 200)
                        cv2.rectangle(check_frame, (tmp_x, tmp_y), (tmp_x + NEG_IMG_WIDTH, tmp_y + NEG_IMG_WIDTH), (0, 0, 255), thickness=2)
                        self.old_neg_dict[tmp_img_name] = [[tmp_x, tmp_y, NEG_IMG_WIDTH, NEG_IMG_WIDTH]]
                        cv2.imwrite(os.path.join(self.check_img_dir, tmp_img_name), check_frame) # save img for check
                        cv2.imwrite(os.path.join(self.work_img_dir, tmp_img_name), org_frame) # save origin img

            if count % 400 == 0:
                print 'now process frame', count, fn

    def save_org_result(self):
        import json
        # save pos box and neg box
        with open("%s/old_pos.txt" % self.work_dir, "w") as f:
            json.dump(self.old_pos_dict, f)
        with open("%s/old_neg.txt" % self.work_dir, "w") as f:
            json.dump(self.old_neg_dict, f)
        with open("%s/new_pos.txt" % self.work_dir, "w") as f:
            json.dump(self.new_pos_dict, f)
        with open("%s/new_neg.txt" % self.work_dir, "w") as f:
            json.dump(self.new_neg_dict, f)


if __name__ == '__main__':
    t = TrainingDataGenerator(PLACEMENT, PROCESS_DATE, OUTPUT_DIR, VIDEO_DIR)
    t.run()
