import numpy as np
import cv2
import os
from sklearn.svm import SVC

WINSIZE = (64, 64)
BLOCKSIZE = (16, 16)
BLOCKSTRIDE = (8, 8)
CELLSIZE = (8, 8)
NBINS = 9


class HogSvmProcessor:
    def __init__(self, img_dir, pos_dict, neg_dict):
        self.img_dir = img_dir
        self.old_pos_dict = pos_dict
        self.old_neg_dict = neg_dict
        self.new_pos_dict = {}
        self.new_neg_dict = {}
        self.key_list = []
        self.hog_worker = cv2.HOGDescriptor(WINSIZE, BLOCKSIZE, BLOCKSTRIDE, CELLSIZE, NBINS)
        self.svm_cls = SVC()

    def run(self):
        self.get_features(self.old_pos_dict, self.old_neg_dict)
        self.svm_train()
        self.adjust_hard_example()
        self.get_features(self.new_pos_dict, self.new_neg_dict)
        self.svm_train()

    def get_features(self, pos_dict, neg_dict):
        self.features = []
        self.response = []
        # prepare pos data
        for k in pos_dict:
            tmp_f_path = os.path.join(self.img_dir, k)
            for b in pos_dict[k]:
                x, y, w, h = b
                frame = cv2.imread(tmp_f_path, 0)
                tmp_frame = frame[y:y + h, x:x + w]
                tmp_frame = cv2.resize(tmp_frame, WINSIZE)
                tmp_hog_feature = self.hog_worker.compute(tmp_frame)
                self.features.append(tmp_hog_feature)
                self.response.append(1)
                self.key_list.append([k, b])  # save file path and box info

        pos_len = len(self.features)
        print 'get pos data', pos_len

        # prepare neg data
        for k in neg_dict:
            tmp_f_path = os.path.join(self.img_dir, k)
            for b in neg_dict[k]:
                x, y, w, h = b
                frame = cv2.imread(tmp_f_path, 0)
                tmp_frame = frame[y:y + h, x:x + w]
                tmp_hog_feature = self.hog_worker.compute(tmp_frame)
                self.features.append(tmp_hog_feature)
                self.response.append(0)
                self.key_list.append([k, b])

        data_len = len(self.features)
        hog_feature_len = len(self.features[0])
        print 'get neg data', len(self.features) - pos_len
        self.features = np.array(self.features)
        # fix this error: ValueError: Found array with dim 3. Estimator expected <= 2.
        self.features = np.reshape(self.features, (data_len, hog_feature_len))
        self.response = np.array(self.response)

    def svm_train(self):
        # train the model and get score
        self.svm_cls.fit(self.features, self.response)
        print 'score:', self.svm_cls.score(self.features, self.response)

    def adjust_hard_example(self):
        test_res = self.svm_cls.predict(self.features)
        data_len = len(test_res)
        for i in range(data_len):
            groundtruth = self.response[i]
            predict_res = test_res[i]
            if groundtruth == 1:  # pos
                if groundtruth == predict_res:  # save right pos in new_pos_dict
                    k, b = self.key_list[i]
                    if k not in self.new_pos_dict:
                        self.new_pos_dict[k] = []
                    self.new_pos_dict[k].append(b)
            else:  # neg
                if groundtruth == predict_res:  # save right neg in new_old_dict
                    k, b = self.key_list[i]
                    if k not in self.new_neg_dict:
                        self.new_neg_dict[k] = []
                    self.new_neg_dict[k].append(b)

    def return_new_res(self):
        return self.new_pos_dict, self.new_neg_dict


if __name__ == '__main__':
    pos = eval(open('/home/jac/Documents/work_tmp/8100224_2016-03-05/old_pos.txt').read())
    neg = eval(open('/home/jac/Documents/work_tmp/8100224_2016-03-05/old_neg.txt').read())
    img_dir = '/home/jac/Documents/work_tmp/8100224_2016-03-05/imgs'
    t = HogSvmProcessor(img_dir, pos, neg)
    t.run()
