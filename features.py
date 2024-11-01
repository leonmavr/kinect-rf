import cv2
import numpy as np


def feature_preprocess(img, quantization_levels=16, resize_factor=0.1):
    new_dims = (int(img.shape[1] * resize_factor), int(img.shape[0] * resize_factor))
    img = cv2.resize(img, new_dims, interpolation=cv2.INTER_NEAREST)
    return (img / (256 / quantization_levels)).astype(np.int16) * (256 // quantization_levels)


def extract_features(img, label_img=None, resize_factor=0.1, mask_size=11):
    assert len(img.shape) == 2  # greyscale img
    # do resizing, quantisation, etc
    img = feature_preprocess(img, resize_factor=resize_factor)
    if label_img is not None: 
        label_img = cv2.resize(label_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    h, w = img.shape
    features, labels = [], []
    # the offset locations inside the mask where features are computed
    hm = mask_size // 2
    qm = mask_size // 4
    offsets = [
                [-hm, -hm], [-hm//2, -hm], [0, -hm]  , [hm//2, -hm], [hm, -hm],
                [-hm, 0]  , [-hm//2, 0]  , [hm//2, 0], [hm, 0] ,
                [-hm, hm] , [-hm//2, hm] , [0, hm]   , [hm//2, hm] , [hm, hm],
                [-qm, -qm], [-qm//2, -qm], [0, -qm]  , [qm//2, -qm], [qm, -qm],
                [-qm, 0]  , [-qm//2, 0]  , [qm//2, 0], [qm, 0] ,
                [-qm, qm] , [-qm//2, qm] , [0, qm]   , [qm//2, qm] , [qm, qm],
              ]
    
    for i in range(mask_size//2, h-mask_size//2):  # avoid boundary pixels for larger offset window
        for j in range(mask_size//2, w-mask_size//2):
            depth_origin = img[i, j]
            depth_diffs = []
            for u, v in offsets:
                ni, nj = i + u, j + v
                # Ensure offsets are within bounds
                if 0 <= ni < h and 0 <= nj < w:
                    depth_diffs.append(img[ni, nj] - depth_origin)
                else:
                    depth_diffs.append(0)  # If out of bounds, use 0 as a fallback

            # combine the depth at origin and depth differences into a feature vector
            feature_vector = np.hstack((depth_origin, depth_diffs))
            features.append(feature_vector)
            if label_img is not None:
                labels.append(label_img[i, j])
    
    return np.array(features), np.array(labels)
