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

def get_features(image, index):
    """Extraction of keypoints and descriptors with black-white image."""
    image_to_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    algorithm = select_algorithm(index=index)
    keypoints, descriptors = algorithm.detectAndCompute(image_to_gray, None)
    return keypoints, descriptors


def matching_features(index, reference_descriptors, test_descriptors):
    if index == 0:
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_SL2)
    elif index == 1 or index == 2:
        matcher = cv.BFMatcher(cv.NORM_HAMMING2)
    keypoints_who_matches = matcher.knnMatch(reference_descriptors, test_descriptors, k=2)
    return keypoints_who_matches

def search_good_match(ratio_test,keypoints_who_matches):
    best_matchs = []
    for m, n in keypoints_who_matches:

        if (m.distance < ratio_test * n.distance):
            best_matchs.append(m)
    return best_matchs

def search_homography(reference_image,test_image,best_matchs,reference_keypoints,test_keypoints):
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



def test_algo(event):
    cv.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv.namedWindow("Logo")
    cv.namedWindow("Trackbar")
    cv.createTrackbar("logo-blur", "Trackbar", 0, 5, nothing)
    cv.createTrackbar("logo-scale", "Trackbar", 100, 100, nothing)
    cv.createTrackbar("webcam-blur", "Trackbar", 0, 5, nothing)
    cv.createTrackbar("ratio-test", "Trackbar", 80, 100, nothing)
    index = lbox.curselection()[0]

    reference_image = cv.imread("./images/hamed4.jpg")  # Importation de l'image en niveau de gris, cv.IMREAD_GRAYSCALE


    reference_keypoints, reference_descriptors = get_features(image=reference_image,index=index)
    reference_image_to_gray = cv.cvtColor(reference_image, cv.COLOR_BGR2GRAY)



    cap = cv.VideoCapture(0)  # Capture des images obtenues par la webcam
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        scale_percent = cv.getTrackbarPos("logo-scale", "Trackbar")  # percent of original size
        width = int(reference_image_to_gray.shape[1] * scale_percent / 100)
        height = int(reference_image_to_gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        reference_image_to_gray = cv.resize(reference_image_to_gray, dim)
        logo_blur_intensity = cv.getTrackbarPos("logo-blur", "Trackbar")
        reference_image_to_gray = cv.GaussianBlur(reference_image_to_gray, (logo_blur_intensity * 2 + 1, logo_blur_intensity * 2 + 1), 0)

        cv2.imshow('Logo', reference_image_to_gray)
        ratio_test = cv2.getTrackbarPos("ratio-test", "Trackbar") / 100

        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        webcam_blur_intensity = cv.getTrackbarPos("webcam-blur", "Trackbar")
        frame = cv.GaussianBlur(frame, (webcam_blur_intensity * 2 + 1, webcam_blur_intensity * 2 + 1), 0)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # points2=akaze.detect(gray ,None)
        test_keypoints, test_descriptors = get_features(image=frame, index=index)

        keypoints_who_matches = matching_features(index=index, reference_descriptors=reference_descriptors, test_descriptors=test_descriptors)

        best_matchs = search_good_match(ratio_test=ratio_test,keypoints_who_matches=keypoints_who_matches)

        test_image_with_draw_keypoints = search_homography(reference_image=reference_image, test_image=frame,
                                                               best_matchs=best_matchs, reference_keypoints=reference_keypoints, test_keypoints=test_keypoints)
        cv.imshow("Homography", test_image_with_draw_keypoints)
            # Display the resulting frame
        if cv.waitKey(1) == ord('q'):
                    break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()




fenetre = Tk()
Nom = ["AKAZE", "BRISK", "ORB"]
lbox = Listbox(fenetre, width=8, height=3, font="Verdana 30 bold", selectbackground="red")
lbox.pack(padx=50, pady=50)

for item in Nom:
    lbox.insert(END, item)

lbox.bind("<<ListboxSelect>>", test_algo)

mainloop()
