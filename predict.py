from features import extract_features, feature_preprocess
import cv2
import numpy as np
import pickle
import sys


def predict(img, clf, resize_factor=0.075, mask_size=11):
    preprocessed_img = feature_preprocess(img, resize_factor=resize_factor)
    h, w = preprocessed_img.shape
    features, _ = extract_features(img, resize_factor=resize_factor, mask_size=mask_size)
    predictions = clf.predict(features)

    img_prediction = np.zeros((h, w), dtype=np.uint8)
    head_intensity = 100
    hand_intensity = 200
    
    # apply predictions to the valid region within the output image
    valid_start = mask_size // 2
    valid_h = h - mask_size + 1
    valid_w = w - mask_size + 1
    img_prediction[valid_start:valid_start + valid_h, valid_start:valid_start + valid_w] = \
        np.where(predictions.reshape(valid_h, valid_w) == 1, head_intensity,
                 np.where(predictions.reshape(valid_h, valid_w) == 2, hand_intensity, 0))
    
    original_dims = (img.shape[1], img.shape[0])
    img_prediction = cv2.resize(img_prediction, original_dims, interpolation=cv2.INTER_NEAREST)
    return img_prediction

