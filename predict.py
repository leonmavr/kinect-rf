import cv2
import numpy as np
import pickle
import sys


def extract_features(img: np.ndarray):
    h, w = img.shape
    features = []
    positions = []
    for i in range(1, h-1):  # avoid boundary pixels
        for j in range(1, w-1):
            neighborhood = img[i-1:i+2, j-1:j+2].flatten()
            depth_value = img[i, j]
            pixel_position = [i, j]
            feature_vector = np.hstack((neighborhood, depth_value, pixel_position))
            features.append(feature_vector)
            positions.append((i, j))
    return np.array(features), positions


def predict_pixels(clf, img_depth: np.ndarray) -> np.ndarray:
    # downsample by 2 because features are computed this way
    resized_image = cv2.resize(img_depth, (img_depth.shape[1] // 2, img_depth.shape[0] // 2))
    # extract features
    X_new, positions = extract_features(img_depth)
    # predict the labels for each pixel
    y_pred = clf.predict(X_new)

    img_labels = np.zeros(img_depth.shape, dtype=np.uint8)
    for idx, (i, j) in enumerate(positions):
        # multiply the class label (1 or 2) with 100 to highlight them
        img_labels[i, j] = y_pred[idx]*100
    return img_labels


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python <this_program.py> <classifier.clf> <depth_image.png>")
        sys.exit(0)

    clf_path = sys.argv[1]
    # import the classifier first
    with open(clf_path, 'rb') as f:
        clf = pickle.load(f)

    img_depth = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    img_predicted = predict_pixels(clf, img_depth)
    cv2.imshow('predicted labels', img_predicted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
