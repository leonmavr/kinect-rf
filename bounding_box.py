import cv2
import numpy as np
from collections import namedtuple
from typing import NamedTuple, List
import sys

BoundingBox = namedtuple('BoundingBox', ['x0', 'y0', 'x1', 'y1'])

def find_bboxes(img: np.ndarray) -> List[NamedTuple]:
    # head and hands bounding boxes
    ret: List[NamedTuple]  = []
    # find the two non-zero intensities in the histogram
    # the lower one will be the label value for the head and the higher for hands
    unique_vals = np.unique(img)
    nonzero_vals = unique_vals[unique_vals > 0]
    if len(nonzero_vals) == 2:
        head_intensity, hand_intensity = sorted(nonzero_vals)
    else:
        raise ValueError("Label image must contain exactly 2 non-zero intensities")
    # create two binary masks - one for the head and one for hands 
    hand_mask = cv2.inRange(img, np.array(hand_intensity),\
        np.array(hand_intensity))
    head_mask = cv2.inRange(img, np.array(head_intensity),\
        np.array(head_intensity))
    # and refine them, e.g. close gaps and smoothen borders
    kernel = np.ones((13, 13), np.uint8)
    hand_closed = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel)
    head_closed = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, kernel)

    hand_contours, _ = cv2.findContours(hand_closed, cv2.RETR_EXTERNAL,\
        cv2.CHAIN_APPROX_SIMPLE)
    head_contours, _ = cv2.findContours(head_closed, cv2.RETR_EXTERNAL,\
        cv2.CHAIN_APPROX_SIMPLE)
    # keep the largest contour for the head and the two largest for the hands
    head_contours = sorted(head_contours, key=cv2.contourArea,\
        reverse=True)[:1]
    hand_contours = sorted(hand_contours, key=cv2.contourArea,\
        reverse=True)[:2]

    # store and return bounding boxes
    for contour in head_contours:
        x, y, w, h = cv2.boundingRect(contour)
        ret.append(BoundingBox(x, y, x+w, y+h)) 
    for contour in hand_contours:
        x, y, w, h = cv2.boundingRect(contour)
        ret.append(BoundingBox(x, y, x+w, y+h)) 
    return ret


def draw_bboxes(img, bboxes, delay=0, show=False) -> np.ndarray:
    if len(img.shape) == 2:
        ret = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        ret = img.copy().copy()
    for i, bbox in enumerate(bboxes):
        x0, y0, x1, y1 = bbox
        if i == 0: # blue for head
            color = (255, 0, 0)
        else: # green for hands
            color = (0, 255, 0)
        cv2.rectangle(ret, (x0, y0), (x1,y1), color, 2)
    if show:
        cv2.imshow("Bounding boxes", ret)
        cv2.waitKey(delay)
    return ret 
