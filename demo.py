from predict import predict
from bounding_box import find_bboxes, draw_bboxes 
import sys
import cv2
import os
import pickle

cap = cv2.VideoCapture(0)

if __name__ == '__main__':
    clf_path = os.path.join('.', 'rf_head_hands.clf')
    with open(clf_path, 'rb') as f:
        clf = pickle.load(f)

    cap = cv2.VideoCapture(os.path.join('depth_test', 'depth_test.mp4'))
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while True:
        valid, img_depth = cap.read()
        img_depth = cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)
        cv2.imshow('pre-recorded video', img_depth)
        img_predicted = predict(img_depth, clf)
        cv2.imshow('predicted labels', img_predicted)
        c = cv2.waitKey(50)
        if c == ord('q') or not valid:
            break
