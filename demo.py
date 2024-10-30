from predict import predict_pixels 
from bounding_box import find_bboxes, draw_bboxes 
import sys
import cv2
import os
import pickle

cap = cv2.VideoCapture(0)

if __name__ == '__main__':
    clf_path = os.path.join('clf', 'rf_head_hands_02.clf')
    with open(clf_path, 'rb') as f:
        clf = pickle.load(f)
    test_dir = 'depth_test'
    for ftest in sorted(os.listdir(test_dir)):
        img_depth = cv2.imread(os.path.join(test_dir, ftest), cv2.IMREAD_GRAYSCALE)
        img_predicted = predict_pixels(clf, img_depth)
        cv2.imshow('predicted labels', img_predicted)
        cv2.waitKey(5000)
    cv2.destroyAllWindows()


