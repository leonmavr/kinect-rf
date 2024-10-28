import cv2
import numpy as np

def extract_features(img, label_img):
    assert(len(img.shape) == 2) # graysclale img
    h, w = img.shape
    features = []
    labels = []
    ### extract the following features:
    # 1. origin
    # 2. depth value at origin
    # 3. values at a grid around the origin
    for i in range(1, h-1):  # avoid boundary pixels
        for j in range(1, w-1):
            grid = img[i-1:i+2, j-1:j+2].flatten()
            depth_origin = img[i, j]
            origin = [i, j]
            feature_vector = np.hstack((grid, depth_origin, origin))
            features.append(feature_vector)
            labels.append(label_img[i, j])
    return np.array(features), np.array(labels)


