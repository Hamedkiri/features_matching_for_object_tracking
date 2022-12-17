import cv2 as cv
import numpy as np
from tkinter import *
from functionbis import nothing, select_algorithm, matching_features, search_good_match, search_homography, get_features
from functions import INDEX_AKAZE, INDEX_BRISK, INDEX_ORB, INDEX_SIFT

NAME_ALGORITHM1 = "AKAZE"
NAME_ALGORITHM2 = "BRISK"
NAME_ALGORITHM3 = "ORB"
NAME_ALGORITHM4 = "SIFT"
NAME_ALGORITHM5 = "SURF"
NAMES_ALGORITHMS = [0,1,2,3]
NAMES_ALGORITHMS[INDEX_AKAZE] = NAME_ALGORITHM1
NAMES_ALGORITHMS[INDEX_BRISK] = NAME_ALGORITHM2
NAMES_ALGORITHMS[INDEX_ORB] = NAME_ALGORITHM3
NAMES_ALGORITHMS[INDEX_SIFT] = NAME_ALGORITHM4

URL_IMAGE = "./images/hamed2.jpg"
"""def readcam():
    cap.set(cv2.CAP_PROP_FPS, 10)
    _, frame = cap.read()
    return frame"""


def view():
    def test_algo(event):
        cv.namedWindow("Frame", cv.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv.namedWindow("Logo")
        cv.namedWindow("Trackbar")
        cv.createTrackbar("logo-blur", "Trackbar", 0, 5, nothing)
        cv.createTrackbar("logo-scale", "Trackbar", 100, 100, nothing)
        cv.createTrackbar("webcam-blur", "Trackbar", 0, 5, nothing)
        cv.createTrackbar("ratio-test", "Trackbar", 80, 100, nothing)
        index = lbox.curselection()[0]

        """reference_image = cv.imread(
            "./images/hamed4.jpg")  # Importation de l'image en niveau de gris, cv.IMREAD_GRAYSCALE
        reference_keypoints, reference_descriptors = get_features(image=reference_image, index=index)
        reference_image_to_gray = cv.cvtColor(reference_image, cv.COLOR_BGR2GRAY)"""

        cap = cv.VideoCapture(0)  # Capture des images obtenues par la webcam
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            reference_image = frame

            reference_keypoints, reference_descriptors = get_features(image=reference_image, index=index)
            reference_image_to_gray = cv.cvtColor(reference_image, cv.COLOR_BGR2GRAY)
            scale_percent = cv.getTrackbarPos("logo-scale", "Trackbar")  # percent of original size
            width = int(reference_image_to_gray.shape[1] * scale_percent / 100)
            height = int(reference_image_to_gray.shape[0] * scale_percent / 100)
            dim = (width, height)
            reference_image_to_gray = cv.resize(reference_image_to_gray, dim)
            logo_blur_intensity = cv.getTrackbarPos("logo-blur", "Trackbar")
            reference_image_to_gray = cv.GaussianBlur(reference_image_to_gray,
                                                      (logo_blur_intensity * 2 + 1, logo_blur_intensity * 2 + 1), 0)

            cv.imshow('Logo', reference_image)
            ratio_test = cv.getTrackbarPos("ratio-test", "Trackbar") / 100


            # if frame is read correctly ret is True
            webcam_blur_intensity = cv.getTrackbarPos("webcam-blur", "Trackbar")
            frame = cv.GaussianBlur(frame, (webcam_blur_intensity * 2 + 1, webcam_blur_intensity * 2 + 1), 0)
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # points2=akaze.detect(gray ,None)
            test_keypoints, test_descriptors = get_features(image=frame, index=index)

            keypoints_who_matches = matching_features(index=index, reference_descriptors=reference_descriptors,
                                                      test_descriptors=test_descriptors)

            best_matchs = search_good_match(index=index,ratio_test=ratio_test, keypoints_who_matches=keypoints_who_matches)

            test_image_with_draw_keypoints = search_homography(reference_image=reference_image, test_image=frame,
                                                               best_matchs=best_matchs,
                                                               reference_keypoints=reference_keypoints,
                                                               test_keypoints=test_keypoints)
            cv.imshow("Homography", test_image_with_draw_keypoints)
            # Display the resulting frame
            if cv.waitKey(1) == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    fenetre = Tk()
    lbox = Listbox(fenetre, width=8, height=3, font="Verdana 30 bold", selectbackground="red")
    lbox.pack(padx=50, pady=50)

    for item in NAMES_ALGORITHMS:
        lbox.insert(END, item)

    lbox.bind("<<ListboxSelect>>", test_algo)

    mainloop()