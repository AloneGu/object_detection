import os
import argparse
import sys
import cv2
import csv
import xml.etree.ElementTree as ET
import re
import json
import numpy as np


def generate_bg(video_file):
    cap = cv2.VideoCapture(video_file)
    # video_length is the total frames' number
    video_length = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    count, bg_cnt = 0, 0
    fgbg = cv2.BackgroundSubtractorMOG()
    bg = np.zeros((cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT), cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), dtype='int32')
    bg_pixel_map = np.zeros((cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT), cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), dtype='int32')
    print bg.shape
    while count < video_length - 1000:
        ret, frame = cap.read()
        count += 1
        fgmask = fgbg.apply(frame)
        bg_fgmask = ~fgmask  # background is 255, foreground is 0
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        curr_bg = frame & bg_fgmask
        # buffer one bg every 75 frames ( 5 seconds)
        if count % 75 == 0:
            print 'add one bg from frame', count
            bg += curr_bg
            bg_pixel_map += (bg_fgmask/255) # bg count for each pixel
            bg_cnt += 1
        if bg_cnt > 100:  # break if got 100 background images
            break
    cap.release()
    bg_1 = np.uint8(bg / bg_pixel_map)
    return bg_1


class HumanDetector(object):
    def __init__(self, xml, feature_type, hog_config, haar_config, video, image, background, org_flag, foreground_flag,
                 fgmask_flag, output_dir):
        self.xml = xml
        self.feature_type = feature_type
        self.hog_config = hog_config
        self.haar_config = haar_config
        self.video = video
        self.image = image
        self.bg_img = background
        self.org_flag = (org_flag == 'True')
        self.foreground_flag = (foreground_flag == 'True')
        self.fgmask_flag = (fgmask_flag == 'True')
        self.output_dir = output_dir

        # normalize file path
        if self.output_dir != None and (not self.output_dir.endswith('/')):
            self.output_dir = self.output_dir + '/'

        self.my_detector = None

        # Initial my detector
        if self.feature_type == 'haar':
            # haar config
            self.haar_scalefactor, self.haar_minneighbor, self.haar_minsize, self.haar_maxsize, self.haar_flags = self.extract_haar_params()
            self.my_detector = cv2.CascadeClassifier(self.xml)
        else:
            # hog config
            WINSIZE, BLOCKSIZE, BLOCKSTRIDE, CELLSIZE, NBINS, self.hog_winstride, self.hog_padding, self.hog_scale, self.hog_finalthreshold = self.extract_hog_params()
            self.my_detector = cv2.HOGDescriptor(WINSIZE, BLOCKSIZE, BLOCKSTRIDE, CELLSIZE, NBINS)
            tree = ET.parse(self.xml)
            root = tree.getroot()
            SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0]
            rho = float(root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text)
            svmvec = [[-float(x)] for x in re.sub('\s+', ' ', SVs.text).strip().split(' ')]
            svmvec.append([rho])
            svmvec = np.array(svmvec)
            self.my_detector.setSVMDetector(svmvec)

    def extract_haar_params(self):
        # get haar parameters
        SCALEFACTOR = 1.1
        MINNEIGHBORS = 5
        MINSIZE = (30, 30)
        MAXSIZE = (96, 96)
        FLAGS = cv2.cv.CV_HAAR_SCALE_IMAGE

        if self.haar_config != None:
            with open(self.haar_config) as f:
                f_content = f.read()
                json_r = json.loads(f_content)

                SCALEFACTOR = float(json_r['SCALEFACTOR'])

                MINNEIGHBORS = int(json_r['MINNEIGHBORS'])

                MINSIZE_STR = json_r['MINSIZE'][1:-1]
                MINSIZE_INT = [int(x) for x in MINSIZE_STR.split(',')]
                MINSIZE = (MINSIZE_INT[0], MINSIZE_INT[1])

                MAXSIZE_STR = json_r['MAXSIZE'][1:-1]
                MAXSIZE_INT = [int(x) for x in MAXSIZE_STR.split(',')]
                MAXSIZE = (MAXSIZE_INT[0], MAXSIZE_INT[1])

                FLAGS = getattr(cv2.cv, json_r['FLAGS'])

        return SCALEFACTOR, MINNEIGHBORS, MINSIZE, MAXSIZE, FLAGS

    def extract_hog_params(self):
        # get hog parameters
        WINSIZE = (96, 96)
        BLOCKSIZE = (16, 16)
        BLOCKSTRIDE = (8, 8)
        CELLSIZE = (8, 8)
        NBINS = 9
        winStride = (8, 8)
        padding = (0, 0)
        scale = 1.01
        finalThreshold = 2

        if self.hog_config != None:
            with open(self.hog_config) as f:
                f_content = f.read()
                json_r = json.loads(f_content)

                WINSIZE_STR = json_r['WINSIZE'][1:-1]
                WINSIZE_INT = [int(x) for x in WINSIZE_STR.split(',')]
                WINSIZE = (WINSIZE_INT[0], WINSIZE_INT[1])

                BLOCKSIZE_STR = json_r['BLOCKSIZE'][1:-1]
                BLOCKSIZE_INT = [int(x) for x in BLOCKSIZE_STR.split(',')]
                BLOCKSIZE = (BLOCKSIZE_INT[0], BLOCKSIZE_INT[1])

                BLOCKSTRIDE_STR = json_r['BLOCKSTRIDE'][1:-1]
                BLOCKSTRIDE_INT = [int(x) for x in BLOCKSTRIDE_STR.split(',')]
                BLOCKSTRIDE = (BLOCKSTRIDE_INT[0], BLOCKSTRIDE_INT[1])

                CELLSIZE_STR = json_r['CELLSIZE'][1:-1]
                CELLSIZE_INT = [int(x) for x in CELLSIZE_STR.split(',')]
                CELLSIZE = (CELLSIZE_INT[0], CELLSIZE_INT[1])

                NBINS = int(json_r['NBINS'])

                winstride_STR = json_r['winStride'][1:-1]
                winstride_INT = [int(x) for x in winstride_STR.split(',')]
                winStride = (winstride_INT[0], winstride_INT[1])

                padding_STR = json_r['padding'][1:-1]
                padding_INT = [int(x) for x in padding_STR.split(',')]
                padding = (padding_INT[0], padding_INT[1])

                scale = float(json_r['scale'])

                finalThreshold = float(json_r['finalThreshold'])

        return WINSIZE, BLOCKSIZE, BLOCKSTRIDE, CELLSIZE, NBINS, winStride, padding, scale, finalThreshold

    def run(self):
        if self.video != None:
            # video mode
            if self.foreground_flag == True and self.bg_img == None:
                self.bg_img = generate_bg(self.video)
                cv2.imshow('background used', self.bg_img)
            cap = cv2.VideoCapture(self.video)
            while (True):
                ret, frame = cap.read()
                if ret == True:
                    self.process_one_img(frame, show_flag=True)
                else:
                    cap.release()
                    break

        elif self.image != None:
            # image mode
            if self.image.endswith('.png') or self.image.endswith('jpg'):  # process one image
                print 'perform human detection on one image'
                self.process_one_img(self.image, True)
            else:  # process images
                # normalize file path
                if not self.image.endswith('/'):
                    self.image = self.image + '/'
                image_lists = os.listdir(self.image)

                # set output directory
                first_img_name = self.image + image_lists[0]
                placement_name = re.findall('\d\d\d\d\d\d\d', first_img_name)[0]
                if self.output_dir != None:
                    csv_output_dir = self.output_dir + 'image_result/csv_result/'
                    img_output_dir = self.output_dir + 'image_result/images/'
                    if not os.path.exists(csv_output_dir):
                        os.makedirs(csv_output_dir)
                    if not os.path.exists(img_output_dir):
                        os.makedirs(img_output_dir)
                    csv_result_file = csv_output_dir + placement_name + '_image_result.csv'
                    csv_w = csv.writer(open(csv_result_file, 'w'))
                    csv_w.writerow(['image name', 'human counts'])

                # start processing images
                for img_name in image_lists:
                    curr_img = self.image + img_name
                    curr_result, image_with_rect = self.process_one_img(curr_img, show_flag=False)
                    print curr_img, 'human count', len(curr_result)
                    if self.output_dir != None:
                        if self.org_flag == True:
                            frame_performed = 'origin_frame'
                        elif self.foreground_flag == True:
                            frame_performed = 'foreground'
                        else:
                            frame_performed = 'fg_mask'
                        # image name to save
                        save_img = img_output_dir + os.path.splitext(img_name)[0] + '_' + self.feature_type + '_' + frame_performed + '_' + str(len(curr_result)) + '.png'
                        cv2.imwrite(save_img, image_with_rect)

                        # write result in csv
                        csv_w.writerow([img_name, len(curr_result)])

                print 'perform human detection on images'  # process images

    def process_one_img(self, img, show_flag):
        results = None
        # video mode will pass image object to this function, image mode just pass the image file path(string) to this function
        if type(img) == str:
            image = cv2.imread(img)
        else:
            image = img

        if self.org_flag == True:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        elif self.foreground_flag == True:
            if self.bg_img != None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # image mode will set the file path(string) of the background image, if there is no background image, video mode will generate one background image object
                if type(self.bg_img) == str:
                    bg = cv2.imread(self.bg_img, 0)
                else:
                    bg = self.bg_img
                gray = gray - bg
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.fgmask_flag:  # actually pass the fgmask image to this function
            gray = image

        if self.feature_type == 'haar':
            # Haar classifier cascade (OpenCV 1.x API only). It can be loaded from XML or YAML file using Load().
            # When the cascade is not needed anymore, release it usingcvReleaseHaarClassifierCascade(&cascade).
            # image : Matrix of the type CV_8U containing an image where objects are detected.objects  Vector of rectangles where each rectangle contains the detected object.
            # scaleFactor : Parameter specifying how much the image size is reduced at each image scale.
            # minNeighbors :Parameter specifying how many neighbors each candidate rectangle should have to retain it.
            # flags : Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.minSize 
            # Minimum possible object size. Objects smaller than that are ignored.maxSize : Maximum possible object size. Objects larger than that are ignored.
            results = self.my_detector.detectMultiScale(
                gray,
                scaleFactor=self.haar_scalefactor,
                minNeighbors=self.haar_minneighbor,
                minSize=self.haar_minsize,
                maxSize=self.haar_maxsize,
                flags=self.haar_flags
            )
        else:
            # win_stride : Window stride. It must be a multiple of block stride.
            # padding : Mock parameter to keep the CPU interface compatibility. It must be (0,0).
            # scale0 : Coefficient of the detection window increase.
            # group_threshold : Coefficient to regulate the similarity threshold. When detected, some objects can be covered by many rectangles. 0 means not to perform grouping. See groupRectangles() .
            results, w = self.my_detector.detectMultiScale(gray, winStride=self.hog_winstride, padding=self.hog_padding,
                                                           scale=self.hog_scale, finalThreshold=self.hog_finalthreshold)

        for (x, y, w, h) in results:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if show_flag:
            cv2.imshow('detection result', image)
            if self.image != None:
                cv2.waitKey(0)
            else:
                cv2.waitKey(30)  # video mode , play like a video
        else:
            return results, image


def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xml', dest='xml')
    parser.add_argument('-f', '--feature', dest='feature_type', default='hog', help='Only support hog or haar feature')
    parser.add_argument('--hog_config', dest='hog_config', default=None)
    parser.add_argument('--haar_config', dest='haar_config', default=None)

    parser.add_argument('-v', '--video', dest='video_file', default=None)
    parser.add_argument('-img', '--image', dest='image_file', default=None)
    parser.add_argument('-b', '--background', dest='bg_image', default=None)

    parser.add_argument('--org_frame', dest='org_flag', default='True')
    parser.add_argument('--foreground', dest='foreground_flag', default='False')
    parser.add_argument('--fgmask', dest='fgmask_flag', default='False')

    parser.add_argument('-o', '--output', dest='output_dir', default=None,
                        help='path of the output directory, None means do not save result')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = my_parse_args()
    print args
    if args.video_file == None and args.image_file == None:
        print ' you need input video or image'
        sys.exit()

    worker = HumanDetector(args.xml, args.feature_type, args.hog_config, args.haar_config, args.video_file,
                           args.image_file, args.bg_image, args.org_flag, args.foreground_flag, args.fgmask_flag,
                           args.output_dir)
    worker.run()