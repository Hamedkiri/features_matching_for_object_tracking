import tkinter as tk
from PIL import Image, ImageTk
import cv2
from functions import nothing, select_algorithm, algorithms_of_matching_features, search_good_match, search_homography_between_images, get_features, random_choice_images
from functions import INDEX_AKAZE, INDEX_BRISK, INDEX_ORB, INDEX_SIFT
from time import strftime


# Create an instance of TKinter Window or frame


class Main_window():

    root = tk.Tk()
    root.title("features-matching-tracking")

    rectangle = []
    mouse_information = []
    mouse_information_to_cropping = []

    reference_blur = [1]
    reference_scale = [1]
    webcam_blur = [1]
    ratio_test = [1]




    index = [0]

    URL_IMAGE = "./images/reference.jpg"
    contain_image = {"reference_image": cv2.imread(URL_IMAGE, cv2.IMREAD_COLOR), "sequence_image": None}







    def __init__(self, side=400):

        root = Main_window.root
        self.root_two = tk.Toplevel(root)
        self.root_two.title("reference image")

        self.side = side

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1100)

        self.photo = None
        self.reference_photo = None

        self.canvas = tk.Canvas(root, width=self.capture.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(row=1, column=1)

        self.canvas_to_resize = tk.Canvas(self.root_two, width=self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                          height=self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas_to_resize.grid(row=1, column=1)

        self.canvas_to_resize.bind('<Double-1>', self.create_and_destroy_rectangle)
        self.canvas_to_resize.bind('<Button-1>', self.moving_rectangle)

        self.snapshot = tk.Button(root, text="Tirer une photo", command=self.snapshot)
        self.snapshot.grid(row=3, column=1)

        self.cropping = tk.Button(self.root_two, text="Rogner l'image", command=self.cropping_of_image)
        self.cropping.grid(row=2, column=1)

        self.mon_menu = tk.Menu(root)
        root.config(menu=self.mon_menu)
        self.algorithmes = tk.Menu(self.mon_menu, tearoff=0)
        self.algorithmes.add_command(label="Akaze", command=self.command_akaze)
        self.algorithmes.add_separator()
        self.algorithmes.add_command(label="Orb", command=self.command_orb)
        self.algorithmes.add_separator()
        self.algorithmes.add_command(label="Brisk", command=self.command_brisk)
        self.algorithmes.add_separator()
        self.algorithmes.add_command(label="Desactive", command=self.command_desactive)
        self.mon_menu.add_cascade(label="Les algorithmes", menu=self.algorithmes)

        """self.curseur_reference_blur = tk.Scale(root, orient="horizontal", command=self.function_reference_blur, from_=0, to=2, label='Blur image_ref')
        self.curseur_reference_blur.set(1)
        self.curseur_reference_blur.grid(row=2, column=2)
        self.curseur_reference_scale = tk.Scale(root, orient="horizontal", command=self.function_reference_scale, from_=0, to=2, label='Scale')
        self.curseur_reference_scale.set(1)
        self.curseur_reference_scale.grid(row=3, column=2)
        self.curseur_webcam_blur = tk.Scale(root, orient="horizontal", command=self.function_webcam_blur, from_=0, to=2, label='Webcam blur')
        self.curseur_webcam_blur.set(1)
        self.curseur_webcam_blur.grid(row=4, column=2)
        self.curseur_ratio_test = tk.Scale(root, orient="horizontal", command=self.function_ratio_test, from_=0, to=2, label='Ratio test')
        self.curseur_ratio_test.set(1)
        self.curseur_ratio_test.grid(row=5, column=2)"""

        self.update_image()

        root.mainloop()

    def get_frame(self):
        """To get image from camera"""
        if self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def update_image(self):
        """To update image on  tkinter window"""
        root = Main_window.root
        contain_image = Main_window.contain_image

        URL_IMAGE = Main_window.URL_IMAGE
        index = Main_window.index
        cv2.namedWindow("Logo")
        cv2.namedWindow("Trackbar")
        cv2.createTrackbar("logo-blur", "Trackbar", 0, 5, nothing)
        cv2.createTrackbar("logo-scale", "Trackbar", 100, 100, nothing)
        cv2.createTrackbar("webcam-blur", "Trackbar", 0, 5, nothing)
        cv2.createTrackbar("ratio-test", "Trackbar", 80, 100, nothing)


        # contain_image["reference_image"] = cv.imread(URL_IMAGE)
        contain_image["sequence_image"] = contain_image["reference_image"]
        #print(contain_image["sequence_image"])

        reference_blur = Main_window.reference_blur
        """reference_scale = Main_window.reference_scale
        webcam_blur = Main_window.webcam_blur
        ratio_test = Main_window.ratio_test"""

        # Get a frame from the video source
        ret, frame = self.get_frame()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")

        try:
            if index[0] != 0:
                reference_keypoints, reference_descriptors = get_features(image=contain_image["sequence_image"],
                                                                          index=index[0])
                reference_image_to_gray = cv2.cvtColor(contain_image["sequence_image"], cv2.COLOR_BGR2GRAY)
                scale_percent = cv2.getTrackbarPos("logo-scale", "Trackbar")  # percent of original size

                width = int(reference_image_to_gray.shape[1] * scale_percent / 100)  # reference_scale[0]
                height = int(reference_image_to_gray.shape[0] * scale_percent/ 100)
                dim = (width, height)
                reference_image_to_gray = cv2.resize(reference_image_to_gray, dim)
                reference_image_to_gray = cv2.GaussianBlur(reference_image_to_gray,
                                                           (int(reference_blur[0]) * 2 + 1,
                                                            int(reference_blur[0]) * 2 + 1),
                                                           0)

                # cv.imshow('Logo', contain_image["sequence_image"])

                # if frame is read correctly ret is True
                # frame = cv2.GaussianBlur(frame, (webcam_blur[0]* 2 + 1, webcam_blur[0] * 2 + 1), 0)
                # try:

                test_keypoints, test_descriptors = get_features(image=frame, index=index[0])

                keypoints_who_matches = algorithms_of_matching_features(index=index[0],
                                                                        reference_descriptors=reference_descriptors,
                                                                        test_descriptors=test_descriptors)

                best_matchs = search_good_match(index=index[0], ratio_test=1,
                                                keypoints_who_matches=keypoints_who_matches)  # ratio_test[0]

                viewimage, test_image_with_draw_keypoints, yes = search_homography_between_images(
                    reference_image=contain_image["sequence_image"], test_image=frame,
                    best_matchs=best_matchs, reference_keypoints=reference_keypoints,
                    test_keypoints=test_keypoints)
                if yes == True:
                    cv2.imwrite(
                        "/home/hamed/My_projects/Machine_Learning/Computer_vision/features_matching/images/images_coffres/test.jpg",
                        viewimage)
                    contain_image["sequence_image"] = viewimage
                    # contain_image["sequence_image"] = contain_image["reference_image"] #random_choice_images([viewimage, contain_image["reference_image"]])

                else:
                    contain_image["sequence_image"] = contain_image[
                        "reference_image"]  # contain_image["reference_image"] #random_choice_images([viewimage, contain_image["reference_image"]])

                    # except:
                    # pass
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(test_image_with_draw_keypoints))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            else:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)


        except:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)





        root.after(5, self.update_image)


    def update_image2(self):
        """To update image on  tkinter window"""

        last_image = cv2.imread("images/frame.jpg", cv2.IMREAD_COLOR)
        last_image = cv2.cvtColor(last_image, cv2.COLOR_BGR2RGB)

        # Get a frame from the video source



        self.reference_photo = ImageTk.PhotoImage(image=Image.fromarray(last_image))
        self.canvas_to_resize.create_image(0, 0, image=self.reference_photo, anchor=tk.NW)

        #root_two.after(20, self.update_image2)

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.get_frame()
        if ret:
            #strftime("%d-%m-%Y-%H-%M-%S") +
            cv2.imwrite("images/"+"frame" + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite("images/"+"reference" + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


        self.update_image2()

    def create_and_destroy_rectangle(self, event):
        rectangle = Main_window.rectangle
        mouse_information = Main_window.mouse_information
        mouse_information_to_cropping = Main_window.mouse_information_to_cropping


        x, y = event.x, event.y

        mouse_information.append(x)
        mouse_information.append(y)

        mouse_information_to_cropping.append(x)
        mouse_information_to_cropping.append(y)


        if len(rectangle) == 0:
            rectangle.append(self.canvas_to_resize.create_rectangle((x-250, y-250), (x+250, y+250), outline='blue'))
        else:
            self.canvas_to_resize.delete(rectangle[0])
            rectangle.pop()


    def moving_rectangle(self,event):
        rectangle = Main_window.rectangle
        mouse_information = Main_window.mouse_information
        mouse_information_to_cropping = Main_window.mouse_information_to_cropping
        x = event.x
        y = event.y



        if (len(mouse_information) * len(mouse_information_to_cropping)) != 0:
            mouse_information_to_cropping[0] = x
            mouse_information_to_cropping[1] = y





            moving_x = x - mouse_information[0]
            moving_y = y - mouse_information[1]

        if len(rectangle) != 0:
            self.canvas_to_resize.move(rectangle[0], moving_x, moving_y)

    def cropping_of_image(self):
        mouse_information_to_cropping = Main_window.mouse_information_to_cropping
        print(mouse_information_to_cropping)

        if(len(mouse_information_to_cropping) != 0):
            x = mouse_information_to_cropping[0]
            y = mouse_information_to_cropping[1]
            image_register = cv2.imread("images/frame.jpg", cv2.IMREAD_COLOR)
            image_register = cv2.cvtColor(image_register, cv2.COLOR_BGR2RGB)

            print((x, y))

            cv2.imwrite("images/"+"reference" + ".jpg", cv2.cvtColor(image_register[y-250:y+250, x-250:x+250], cv2.COLOR_RGB2BGR))

    def command_akaze(self):
        index = Main_window.index
        index[0] = INDEX_AKAZE

    def command_brisk(self):
        index = Main_window.index
        index[0] = INDEX_BRISK

    def command_orb(self):
        index = Main_window.index
        index[0] = INDEX_ORB

    def command_desactive(self):
        index = Main_window.index
        index[0] = 0

    def function_reference_blur(self, blur):
        reference_blur = Main_window.reference_blur
        reference_blur[0] = blur

        #reference_blur[0] = (1/100)*float(blur)

    def function_reference_scale(self, scale):
        reference_scale = Main_window.reference_scale
        print(scale)
        reference_scale[0] = scale

        #reference_scale[0] = (1/100)*float(scale)

    def function_webcam_blur(self, webcam):
        webcam_blur = Main_window.webcam_blur
        webcam_blur[0] = webcam

        #webcam_blur[0] = (1/100)*float(webcam)

    def function_ratio_test(self, ratio):
        ratio_test = Main_window.ratio_test
        ratio_test[0] = ratio
        #ratio_test[0] = (1/100)*float(ratio)
















Main_window(side=1000)


