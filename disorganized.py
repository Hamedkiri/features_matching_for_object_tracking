import cv2
from tkinter import *
import numpy as np




def algorithm_sift(url_image):
    sift_algorithm = cv2.SIFT_create()
    image = cv2.imread(url_image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift_algorithm.detectAndCompute(gray_image, None)

    return keypoints, descriptors[1]



def algorithm_akazeur(url_image):
    akaze = cv2.AKAZE_create()
    image = cv2.imread(url_image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = akaze.detect(gray_image, None)
    descriptors = akaze.compute(gray_image, keypoints)

    return keypoints, descriptors[1]


def algorithm_orbeur(url_image):
    orb = cv2.ORB_create(2000)
    image = cv2.imread(url_image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = orb.detect(gray_image, None)
    descriptors = orb.compute(image, keypoints)

    return keypoints, descriptors[1]


def algorithm_briskeur(url_image):
    brisk = cv2.BRISK_create()
    image = cv2.imread(url_image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = brisk.detect(image, None)
    descriptors = brisk.compute(image, keypoints)

    return keypoints, descriptors[1]




def nothing(x):
    pass




def create_detector(index):
    if index == 0:
        return algorithm_akazeur
    elif index == 1:
        return algorithm_orbeur
    elif index == 2:
        return algorithm_briskeur
    elif index == 3:
        return algorithm_sift



def test_algo(event):
    # index=lbox.curselection()[0]
    # cap = cv2.VideoCapture(0)
    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow("Logo")
    cv2.namedWindow("Trackbar")
    cv2.createTrackbar("logo-blur", "Trackbar", 0, 5, nothing)
    cv2.createTrackbar("logo-scale", "Trackbar", 100, 100, nothing)
    cv2.createTrackbar("webcam-blur", "Trackbar", 0, 5, nothing)
    cv2.createTrackbar("ratio-test", "Trackbar", 80, 100, nothing)

    img = cv2.imread("Arcure-_logo_2.png", cv2.IMREAD_GRAYSCALE)  # Importation de l'image en niveau de gris
    index = lbox.curselection()[0]
    detector = create_detector(index)
    points, desc_image = detector(img)
    if index == 0:
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_SL2)
    elif index == 1 or index == 2:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
    cap = cv2.VideoCapture(0)  # Capture des images obtenues par la webcam
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        scale_percent = cv2.getTrackbarPos("logo-scale", "Trackbar")  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim)
        logo_blur_intensity = cv2.getTrackbarPos("logo-blur", "Trackbar")
        img = cv2.GaussianBlur(img, (logo_blur_intensity * 2 + 1, logo_blur_intensity * 2 + 1), 0)

        cv2.imshow('Logo', img)
        ratio_test = cv2.getTrackbarPos("ratio-test", "Trackbar") / 100

        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        webcam_blur_intensity = cv2.getTrackbarPos("webcam-blur", "Trackbar")
        frame = cv2.GaussianBlur(frame, (webcam_blur_intensity * 2 + 1, webcam_blur_intensity * 2 + 1), 0)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            h, w = img.shape

            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            # for x,y in np.array(img):
            # cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            dst = cv2.perspectiveTransform(pts, M)

            oui = []
            for a in good_points:
                oui.append(points2[a.trainIdx])

            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            pte = cv2.drawKeypoints(homography, oui, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # hshs
            # cv2.rectangle(img, (110, 200), (200, 250), (255, 0, 0), 1)

            cv2.imshow("Homography", pte)

            # Display the resulting frame
            # cv.imshow('frame', gray)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def test_algo(event):
    fenetre = Tk()
    Nom = ["AKAZE", "BRISK", "ORB", "SIFT"]
    lbox = Listbox(fenetre, width=8, height=3, font="Verdana 30 bold", selectbackground="red")
    lbox.pack(padx=50, pady=50)
    for item in Nom:
        lbox.insert(END, item)
    lbox.bind("<<ListboxSelect>>", test_algo)
    mainloop()

