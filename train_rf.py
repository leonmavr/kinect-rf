from features import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import pickle

### Instructions
# 1. Store your sorted by name depth images in directory `depth` as depth_%06d.png or depth_%06d.jpg
# 2. Store your sorted by name labelled images in directory `labelled` as lbl_%06d.png or lbl_%06d.jpg
# 3. Run this script. It will save the classifier as a .pkl file

dir_train = 'depth_train'
dir_labels = 'labelled' 

if __name__ == '__main__':
    X, y = [], []
    for fdepth, flabel in zip(sorted(os.listdir(dir_train)),\
                              sorted(os.listdir(dir_labels))):
        depth = cv2.imread(os.path.join(dir_train, fdepth), cv2.IMREAD_GRAYSCALE)
        lbl = cv2.imread(os.path.join(dir_labels, flabel), cv2.IMREAD_GRAYSCALE)
        XX, yy = extract_features(depth, lbl)
        X.append(XX)
        y.append(yy)

    X = np.vstack(X) # stack into 2D array where each row is a pixel's feature vector
    y = np.hstack(y) # flatten labels into a column vector 

    # split dataset in training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y,\
        test_size=0.2, random_state=42)
    # training
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # evaluate performance
    accuracy = clf.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # dump trained classifier file
    clf_file = 'rf_head_hands.clf'
    with open(clf_file, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Classifier saved as {clf_file}")
