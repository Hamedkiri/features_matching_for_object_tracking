from tkinter import *
import cv2 as cv
import numpy as np

import cv2



INDEX_AKAZE = 0
INDEX_BRISK = 1
INDEX_ORB = 2
INDEX_SIFT = 3
INDEX_SURF = 4

URL_IMAGE = "./images/SA.jpg"
URL_IMAGE2 = "./images/hamed.jpg"


def nothing(x):
    pass


def select_algorithm(index):
    """To select different algorithm of features-matching."""
    if index == INDEX_AKAZE:
        # default threshold=0.00001, nOctaves=16
        return cv.AKAZE_create()
    elif index == INDEX_BRISK:
        return cv.BRISK_create()
    elif index == INDEX_ORB:
        # default nfeatures=10000, scaleFactor=1.2
        return cv.ORB_create(2000)
    elif index == INDEX_SIFT:
        return cv.SIFT_create()
    elif index == INDEX_SIFT:
        return cv2.SIFT_create()

def get_features(image, index):
    """Extraction of keypoints and descriptors with black-white image."""
    image_to_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    algorithm = select_algorithm(index=index)
    keypoints, descriptors = algorithm.detectAndCompute(image_to_gray, None)
    return keypoints, descriptors


def matching_features(index, reference_descriptors, test_descriptors):
    """Matching between the descriptors"""
    if index == INDEX_AKAZE:
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_SL2)
    elif index == INDEX_BRISK  or index == INDEX_ORB:
        matcher = cv.BFMatcher(cv.NORM_HAMMING2)
    elif index == INDEX_SIFT:
        matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

    if index != INDEX_SIFT:
        keypoints_who_matches = matcher.knnMatch(reference_descriptors, test_descriptors, k=2)
    else:
        keypoints_who_matches = matcher.match(reference_descriptors, test_descriptors)

    return keypoints_who_matches

def search_good_match(index,ratio_test,keypoints_who_matches):
    """To select in the key points those corresponding to a lower distance ratio_test"""

    #print(keypoints_who_matches)
    best_matchs = []
    if index != INDEX_SIFT:
        for m, n in keypoints_who_matches:
            if (m.distance < ratio_test * n.distance):
                best_matchs.append(m)
    else:
        for m in keypoints_who_matches:
            for n in keypoints_who_matches:
                if (m.distance < ratio_test * n.distance):
                    best_matchs.append(m)

    return best_matchs

def search_homography(reference_image,test_image,best_matchs,reference_keypoints,test_keypoints):
    """ To search homography between the two images """

    reference_image_to_gray = cv.cvtColor(reference_image, cv.COLOR_BGR2GRAY)
    try:
        if len(best_matchs) >= 4:
            src_pts = np.float32([reference_keypoints[m.queryIdx].pt for m in best_matchs]).reshape(-1, 1, 2)
            dst_pts = np.float32([test_keypoints[m.trainIdx].pt for m in best_matchs]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            h, w = reference_image_to_gray.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            if M is not None:
                dst = cv.perspectiveTransform(pts, M)
                oui = []
                for a in best_matchs:
                    oui.append(test_keypoints[a.trainIdx])

                    image_with_homography = cv.polylines(test_image, [np.int32(dst)], True, (255, 0, 0), 3)
                    test_image_with_draw_keypoints = cv.drawKeypoints(image_with_homography, best_matchs, None,
                                                                      flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    return test_image_with_draw_keypoints
            else:
                return test_image
        else:
            return test_image
    except:
        return test_image