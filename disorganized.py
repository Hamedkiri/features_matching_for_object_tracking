from tkinter import *
import cv2 as cv
import numpy as np

import cv2


def nothing(x):
    pass


def akazeur(image):
    akaze = cv.AKAZE_create()
    points = akaze.detect(image, None)
    descripteurs = akaze.compute(image, points)

    return points, descripteurs[1]


def orbeur(image):
    orb = cv.ORB_create(2000)
    points = orb.detect(image, None)
    descripteurs = orb.compute(image, points)

    return points, descripteurs[1]


def briskeur(image):
    brisk = cv.BRISK_create()
    points = brisk.detect(image, None)
    descripteurs = brisk.compute(image, points)

    return points, descripteurs[1]


def create_detector(index):
    if index == 0:
        return akazeur
    elif index == 1:
        return briskeur
    elif index == 2:
        return orbeur


def test_algo(event):
    # index=lbox.curselection()[0]
    # cap = cv2.VideoCapture(0)
    cv.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv.namedWindow("Logo")
    cv.namedWindow("Trackbar")
    cv.createTrackbar("logo-blur", "Trackbar", 0, 5, nothing)
    cv.createTrackbar("logo-scale", "Trackbar", 100, 100, nothing)
    cv.createTrackbar("webcam-blur", "Trackbar", 0, 5, nothing)
    cv.createTrackbar("ratio-test", "Trackbar", 80, 100, nothing)

    img = cv.imread("./images/hamed.jpg", cv.IMREAD_GRAYSCALE)  # Importation de l'image en niveau de gris
    index = lbox.curselection()[0]
    detector = create_detector(index)
    points, desc_image = detector(img)
    if index == 0:
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_SL2)
    elif index == 1 or index == 2:
        matcher = cv.BFMatcher(cv.NORM_HAMMING2)
    cap = cv.VideoCapture(0)  # Capture des images obtenues par la webcam
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        scale_percent = cv.getTrackbarPos("logo-scale", "Trackbar")  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv.resize(img, dim)
        logo_blur_intensity = cv.getTrackbarPos("logo-blur", "Trackbar")
        img = cv.GaussianBlur(img, (logo_blur_intensity * 2 + 1, logo_blur_intensity * 2 + 1), 0)

        cv2.imshow('Logo', img)
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
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # points2=akaze.detect(gray ,None)
        points2, desc_grayframe = detector(gray)

        matches = matcher.knnMatch(desc_image, desc_grayframe, k=2)

        good_points = []
        for m, n in matches:

            if (m.distance < 1 * n.distance):
                good_points.append(m)

        # query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        # train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        # atrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
        if len(good_points) >= 4:
            src_pts = np.float32([points[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            dst_pts = np.float32([points2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            h, w = img.shape

            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            # for x,y in np.array(img):
            # cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            dst = cv.perspectiveTransform(pts, M)

            oui = []
            for a in good_points:
                oui.append(points2[a.trainIdx])

            homography = cv.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            pte = cv.drawKeypoints(homography, oui, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # hshs
            # cv2.rectangle(img, (110, 200), (200, 250), (255, 0, 0), 1)

            cv.imshow("Homography", pte)

            # Display the resulting frame
            # cv.imshow('frame', gray)
            if cv.waitKey(1) == ord('q'):
                break
        else:
            cv.imshow('frame', gray)
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
