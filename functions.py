import cv2
import numpy as np

INDEX_AKAZE = 0
INDEX_BRISK = 1
INDEX_ORB = 2
INDEX_SIFT = 3
INDEX_SURF = 4

URL_IMAGE = "./images/SA.jpg"
URL_IMAGE2 = "./images/hamed.jpg"
def nothing(x):
    pass


def create_detector(index):

    """To select different algorithm of features-matching."""

    if index == INDEX_AKAZE:
        # default threshold=0.00001, nOctaves=16
        return cv2.AKAZE_create()
    elif index == INDEX_BRISK:
        return cv2.BRISK_create()
    elif index == INDEX_ORB:
        # default nfeatures=10000, scaleFactor=1.2
        return cv2.ORB_create()
    elif index == INDEX_SIFT:
        return cv2.SIFT_create()
    #elif index == INDEX_SURF:
        #return cv2.xfeatures2d.SURF_create(400)


def get_features(img, index):

    """Extraction of key points and descriptors with black-white image."""

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = create_detector(index=index)
    kps, descs = detector.detectAndCompute(gray_image, None)
    return kps, descs, gray_image.shape





def matching_features(ref_features, train_features, index, ratio_test):#img, train_features,

    """Matching descriptors between images."""

    train_kps, train_descs, shape = train_features
    # get features from input image
    ref_kps, ref_descs, ref_shape = ref_features  # ,octave=octave , octave=octave, octave_echelle=octave_echelle
    # check if keypoints are extracted
    if not ref_kps:
        print("No keypoints extracted")
        return None, None

    # now we need to find matching keypoints in 2 sets of descriptors (from sample image and from current image)
    # knnMatch uses k-nearest neighbours algorithm for that
    # bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_SL2)
    if index == INDEX_AKAZE:
        bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_SL2)
    elif index == INDEX_BRISK or index == INDEX_ORB:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2)
    elif index == INDEX_SIFT:
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    elif index == INDEX_SURF:
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    if index != INDEX_SIFT:
        matches = bf.knnMatch(train_descs, ref_descs, k=2)
    else:
        matches = bf.match(train_descs, ref_descs)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matches = bf.knnMatch(train_descs, descs, k=2)

    kps_who_matchs = []
    # apply ratio test to matches of each keypoint
    # idea is if train KP have a matching KP on image, it will be much closer than next closest non-matching KP,
    # otherwise, all KPs will be almost equally far
    try:
        for m, n in matches:
            if m.distance < ratio_test * n.distance:
                kps_who_matchs.append([m])

        print("len good: ", len(kps_who_matchs))
        print("len train kpts ", len(train_kps))

        #print(kps_who_matchs[0][0].trainIdx)
        #print(kps_who_matchs[0][0].queryIdx)


        if len(kps_who_matchs) >= 4:
               # estimate a transformation matrix which maps keypoints from train image coordinates to sample image.
               src_pts = np.float32([ref_kps[m[0].queryIdx].pt for m in kps_who_matchs]).reshape(-1, 1, 2)
               dst_pts = np.float32([train_kps[m[0].trainIdx].pt for m in kps_who_matchs]).reshape(-1, 1, 2)
               M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

               h, w = ref_shape
               if M is not None:
                   pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                   dst = cv2.perspectiveTransform(pts, M)
                   index_good_keys = []
                   for elt in kps_who_matchs:
                       index_good_keys.append(train_kps[elt[0].trainIdx])
                   return dst, index_good_keys
               else:
                   return None, train_kps
        else:
            return None, train_kps
    except:
        pass
        return None, train_kps

"""img = cv2.imread(URL_IMAGE)
img2 = cv2.imread(URL_IMAGE2)
train_features = get_features(img, index=1)
ref_features = get_features(img, index=1)
matching_features(ref_features, train_features, index=1, ratio_test=0.8)"""