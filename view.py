import cv2
import numpy as np
from tkinter import *
from functions import nothing, get_features, detect_features


def view():
    def test_algo(event):
        index = lbox.curselection()[0]
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.namedWindow("Logo")
        cv2.namedWindow("Trackbar")
        cv2.createTrackbar("logo-blur", "Trackbar", 0, 5, nothing)
        cv2.createTrackbar("logo-scale", "Trackbar", 100, 100, nothing)
        cv2.createTrackbar("webcam-blur", "Trackbar", 0, 5, nothing)
        cv2.createTrackbar("ratio-test", "Trackbar", 80, 100, nothing)
        cv2.createTrackbar("AKAZE_seuil", "Trackbar", 1, 2, nothing)
        cv2.createTrackbar("AKAZE_octave_echelle", "Trackbar", 8, 32, nothing)
        cv2.createTrackbar("AKAZE_octave", "Trackbar", 8, 32, nothing)
        while True:
            # get train features

            img = cv2.imread("./images/hamed.jpg")
            # img = cv2.imread("dessin.png")
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            # img = cv2.imread("dessin.png", cv2.IMREAD_UNCHANGED)
            # img = img[:,:,:3]

            scale_percent = cv2.getTrackbarPos("logo-scale", "Trackbar")  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim)

            logo_blur_intensity = cv2.getTrackbarPos("logo-blur", "Trackbar")
            img = cv2.GaussianBlur(img, (logo_blur_intensity * 2 + 1, logo_blur_intensity * 2 + 1), 0)

            cv2.imshow('Logo', img)
            ratio_test = cv2.getTrackbarPos("ratio-test", "Trackbar") / 100
            nombre_point = cv2.getTrackbarPos("AKAZE_seuil", "Trackbar")
            octave = cv2.getTrackbarPos("AKAZE_octave", "Trackbar")
            octave_echelle = cv2.getTrackbarPos("AKAZE_octave_echelle", "Trackbar")
            print("Getting features from train image ...")
            train_features = get_features(img, index, nombre_point, octave, octave_echelle)  # ,octave=octave
            print("train features:", len(train_features[0]))

            _, frame = cap.read()
            webcam_blur_intensity = cv2.getTrackbarPos("webcam-blur", "Trackbar")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frame = cv2.GaussianBlur(frame, (webcam_blur_intensity * 2 + 1, webcam_blur_intensity * 2 + 1), 0)

            # detect features on test image
            print("Detecting features in test image ...")
            region, kps = detect_features(frame, train_features, index, ratio_test=ratio_test,
                                          nombre_point=nombre_point, octave=octave,
                                          octave_echelle=octave_echelle)  # ,octave=octave
            frame = cv2.drawKeypoints(frame, kps, np.array([]), color=(255, 0, 0))
            if region is not None:
                print("Logo detected ...")
                # draw rotated bounding box
                box = cv2.boxPoints(region)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
            else:
                print("No logo detected ...")
            if index == 0:
                cv2.imshow('AKAZE', frame)
            elif index == 1:
                cv2.imshow('BRISK', frame)
            elif index == 2:
                cv2.imshow('ORB', frame)
            elif index == 3:
                cv2.imshow('SIFT', frame)

            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    fenetre = Tk()
    Nom = ["AKAZE", "BRISK", "ORB"]
    lbox = Listbox(fenetre, width=8, height=3, font="Verdana 30 bold", selectbackground="red")
    lbox.pack(padx=50, pady=51)

    for item in Nom:
        lbox.insert(END, item)

    lbox.bind("<<ListboxSelect>>", test_algo)

    mainloop()
