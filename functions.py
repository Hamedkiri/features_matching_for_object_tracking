import cv2
import numpy as np
from tkinter import *
import pandas as pd


def nothing(x):
    pass


def create_detector(nombre_point, index, octave, octave_echelle):  # ,octave
    if index == 0:
        # default threshold=0.00001, nOctaves=16
        return cv2.AKAZE_create()
    elif index == 1:
        return cv2.BRISK_create()
    elif index == 2:
        # default nfeatures=10000, scaleFactor=1.2
        return cv2.ORB_create()
    elif index == 3:
        return cv2.SIFT_create()


def get_features(img, index, nombre_point, octave, octave_echelle):  # ,octave
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = create_detector(index, nombre_point, octave, octave_echelle)  # ,octave
    kps, descs = detector.detectAndCompute(gray, None)
    return kps, descs, img.shape[:2][::-1]





def detect_features(img, train_features, index, ratio_test, nombre_point, octave, octave_echelle):  # ,octave
    train_kps, train_descs, shape = train_features
    # get features from input image
    kps, descs, _ = get_features(img, index, nombre_point, octave, octave_echelle)  # ,octave=octave
    # kps,descs=selectionneur_(img,liste_image_train,1,index,nombre_point,octave,octave_echelle)
    # check if keypoints are extracted
    if not kps:
        print("No keypoints extracted")
        return None, None

    # now we need to find matching keypoints in 2 sets of descriptors (from sample image and from current image)
    # knnMatch uses k-nearest neighbours algorithm for that
    # bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_SL2)
    if index == 0:
        bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_SL2)
    elif index == 1 or index == 2:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2)
    matches = bf.knnMatch(train_descs, descs, k=2)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matches = bf.knnMatch(train_descs, descs, k=2)

    good = []
    # apply ratio test to matches of each keypoint
    # idea is if train KP have a matching KP on image, it will be much closer than next closest non-matching KP,
    # otherwise, all KPs will be almost equally far
    try:
        for m, n in matches:
            if m.distance < ratio_test * n.distance:
                good.append([m])

        print("len good: ", len(good))
        print("len train kpts ", len(train_kps))

        # stop if we didn't find enough matching keypoints
        if len(good) < 0.1 * len(train_kps):
            print("Not enough matching keypoints")
            return None, kps

        # estimate a transformation matrix which maps keypoints from train image coordinates to sample image
        src_pts = np.float32([train_kps[m[0].queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[m[0].trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)

        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if m is not None:
            # apply perspective transform to train image corners to get a bounding box coordinates on a sample image
            scene_points = cv2.perspectiveTransform(
                np.float32([(0, 0), (0, shape[0] - 1), (shape[1] - 1, shape[0] - 1), (shape[1] - 1, 0)]).reshape(-1, 1,
                                                                                                                 2), m)
            rect = cv2.minAreaRect(scene_points)

            # check resulting rect ratio knowing we have almost square train image
            print("rect11: ", rect[1][1])
            if rect[1][1] > 0:  # and 0.5 < (rect[1][0] / rect[1][1]) < 1.5:
                return rect, kps
    except:
        pass
    return None, kps






# -----------------------------------------------------------------------------------------------2











