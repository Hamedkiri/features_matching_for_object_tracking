import tkinter as tk
from PIL import Image, ImageTk
import cv2
from functions import algorithms_of_matching_features, search_good_match, search_homography_between_images, get_features
from functions import INDEX
from calibration import Calibration_of_camera
from to_test import register_video, write_video

# Create an instance of TKinter Window or frame


class Main_window():

    """Interface of algorithm : Showing object tracking and manipulation of parameters"""

    # Principal window
    root = tk.Tk()
    root.title("features-matching-tracking")

    # Variables to crop reference image
    rectangle = []
    mouse_information = []
    mouse_information_to_cropping = []

    # Variables to run the algorithm
    """reference_blur = [1]
    reference_scale = [1]
    webcam_blur = [1]
    ratio_test = [1]"""

    # Index to choose features-matching algorithm, 0 corresponding disable
    index = [0]

    URL_IMAGE = "./images/reference.jpg"
    contain_image = {"reference_image": cv2.imread(URL_IMAGE, cv2.IMREAD_COLOR),
                     "sequence_image": cv2.imread(URL_IMAGE, cv2.IMREAD_COLOR)}


    def __init__(self, side=1100):

        root = Main_window.root

        # Second window to choose reference image
        self.root_two = tk.Toplevel(root)
        self.root_two.title("reference image")

        self.root_show_reference = tk.Toplevel(root)
        self.root_show_reference.title("Monter l'image de reference")

        self.window_calibration = None

        # Window size
        self.side = side

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.side)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.side)

        # Get video metadata
        video_fps = self.capture.get(cv2.CAP_PROP_FPS),
        height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)

        # we are using x264 codec for mp4
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter('./videos/OUTPUT_PATH_new_meth.mp4', apiPreference=0, fourcc=fourcc,
                                 fps=video_fps[0], frameSize=(int(width), int(height)))


        self.photo = None
        self.reference_photo = None

        # Principal canvas where the algorithm runs
        self.canvas = tk.Canvas(root, width=self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(row=1, column=1)

        # Canvas to crop image
        self.canvas_to_resize = tk.Canvas(self.root_two, width=self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                          height=self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas_to_resize.grid(row=1, column=1)

        # Canvas to show reference image
        self.canvas_show_image = tk.Canvas(self.root_show_reference, width=self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                           height=self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas_show_image.grid(row=1, column=1)

        self.canvas_to_resize.bind('<Double-1>', self.create_and_destroy_rectangle)
        self.canvas_to_resize.bind('<Button-1>', self.moving_rectangle)

        # Button to make a photo
        self.snapshot = tk.Button(root, text="Tirer une photo", command=self.snapshot)
        self.snapshot.grid(row=4, column=1)

        # Button to crop image
        self.cropping = tk.Button(self.root_two, text="Rogner l'image", command=self.cropping_of_image)
        self.cropping.grid(row=2, column=1)

        # Menu to choose algorithm
        self.mon_menu = tk.Menu(root)
        root.config(menu=self.mon_menu)
        self.algorithmes = tk.Menu(self.mon_menu, tearoff=0)
        self.algorithmes.add_command(label="Akaze", command=self.command_akaze)
        self.algorithmes.add_separator()
        self.algorithmes.add_command(label="Orb", command=self.command_orb)
        self.algorithmes.add_separator()
        self.algorithmes.add_command(label="Desactive", command=self.command_desactive)
        self.mon_menu.add_cascade(label="Les algorithmes", menu=self.algorithmes)

        # Variable to know if tracking is active
        self.know_tracking_activate = tk.IntVar()
        self.active_tracking = tk.Checkbutton(root, text="Activer le tracking", variable=self.know_tracking_activate)
        self.active_tracking.grid(row=3, column=1)

        # Variables to run the algorithm
        self.scale_percent = 1
        self.logo_blur_intensity = 0
        self.ratio_test = 0.8

        # To make calibration
        self.active_calibration = False

        # Button to active calibration
        self.Do_calibration = tk.Button(root, text="Calibration photos", command=self.switch_to_calibration)
        self.Do_calibration.grid(row=5, column=1)

        self.text_calibration = ""
        self.counter_to_calibration = 0

        # Function to run video
        self.update_image()

        root.mainloop()



        cv2.destroyAllWindows()

    def get_frame(self):

        """To get image from camera"""

        if self.capture.isOpened():
            ret, frame = self.capture.read()

            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return ret, None

    def application_of_algorithm(self, frame, image, index=[0]):

        """Application of features-matching algorithm"""

        contain_image = Main_window.contain_image

        contain_image["reference_image"] = cv2.imread(Main_window.URL_IMAGE, cv2.IMREAD_COLOR)

        image_sequence = [image]

        try:
            if index[0] != 0:
                # Showing reference image change
                self.update_show_reference()

                # Extraction of keypoints and descriptors in reference image
                reference_keypoints, reference_descriptors = get_features(image=image_sequence[0],
                                                                          index=index[0])
                # Extraction of keypoints and descriptors in camera image
                test_keypoints, test_descriptors = get_features(image=frame, index=index[0])

                # Matching of keypoints by their descriptors
                keypoints_who_matches = algorithms_of_matching_features(index=index[0],
                                                                        reference_descriptors=reference_descriptors,
                                                                        test_descriptors=test_descriptors)

                # Selection of best match
                best_matchs = search_good_match(index=index[0], ratio_test=self.ratio_test,
                                                keypoints_who_matches=keypoints_who_matches)  # ratio_test[0]

                # Searching of homography
                viewimage, test_image_with_draw_keypoints, yes = search_homography_between_images(
                    reference_image=image_sequence[0], test_image=frame,
                    best_matchs=best_matchs, reference_keypoints=reference_keypoints,
                    test_keypoints=test_keypoints)

                if yes == True:
                    # To know if homography is rectangle
                    # If True change reference image by image detected when tracking is active
                    if self.know_tracking_activate.get() == 1:
                        contain_image["sequence_image"] = viewimage
                else:
                    # If False, takes the cropped image as the reference image
                    contain_image["sequence_image"] = contain_image["reference_image"]

                return test_image_with_draw_keypoints


            else:
                return None


        except:

            return None

    def update_image(self):

        """To update tkinter frame image"""

        root = Main_window.root
        index = Main_window.index
        contain_image = Main_window.contain_image

        # Get a frame from the video source
        ret, frame = self.get_frame()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")

        image_back = self.application_of_algorithm(frame=frame, image=contain_image["sequence_image"], index=index)


        if image_back is not None:

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(image_back))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            image_back = cv2.cvtColor(image_back, cv2.COLOR_RGB2BGR)
            #print(image_back.shape)
            self.writer.write(image_back)  # register video



        else:

            WIDTH = 400
            HEIGHT = 300
            C = (WIDTH // 2, HEIGHT // 2)

            contain_image["sequence_image"] = contain_image["reference_image"]
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            if self.active_calibration:
                text = self.canvas.create_text(C, anchor=tk.W, text=self.text_calibration, fill="blue")
                self.canvas.after(1, self.wipe_off, text)
            #write_video(file_path="./videos/test.mp4", frames=register_video, fps=24, side=self.side)


        root.after(1, self.update_image)

    def update_show_reference(self):

        """Function to show reference image"""

        contain_image = Main_window.contain_image

        self.reference_photo = ImageTk.PhotoImage(image=Image.fromarray(contain_image["sequence_image"]))
        self.canvas_show_image.create_image(0, 0, image=self.reference_photo, anchor=tk.NW)

    def update_show_snapshot(self):

        """Show snapshot image"""

        last_image = cv2.imread("images/frame.jpg", cv2.IMREAD_COLOR)
        last_image = cv2.cvtColor(last_image, cv2.COLOR_BGR2RGB)

        # Get a frame from the video source

        self.reference_photo = ImageTk.PhotoImage(image=Image.fromarray(last_image))
        self.canvas_to_resize.create_image(0, 0, image=self.reference_photo, anchor=tk.NW)

    def snapshot(self):

        """To make snapshot and register snapshotting image"""

        ret, frame = self.get_frame()
        if ret:
            cv2.imwrite("images/" + "frame" + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite("images/" + "reference" + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        self.update_show_snapshot()

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
            rectangle.append(
                self.canvas_to_resize.create_rectangle((x - 250, y - 250), (x + 250, y + 250), outline='blue'))
        else:
            self.canvas_to_resize.delete(rectangle[0])
            rectangle.pop()

    def moving_rectangle(self, event):

        """Crop rectangle moving"""

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

        """Mousse information to crop"""

        mouse_information_to_cropping = Main_window.mouse_information_to_cropping
        #print(mouse_information_to_cropping)

        if (len(mouse_information_to_cropping) != 0):
            x = mouse_information_to_cropping[0]
            y = mouse_information_to_cropping[1]
            image_register = cv2.imread("images/frame.jpg", cv2.IMREAD_COLOR)
            image_register = cv2.cvtColor(image_register, cv2.COLOR_BGR2RGB)

            # print((x, y))

            cv2.imwrite("images/" + "reference" + ".jpg",
                        cv2.cvtColor(image_register[y - 250:y + 250, x - 250:x + 250], cv2.COLOR_RGB2BGR))

    # Command to choose algorithm : AKAZE, ORB or disable
    def command_akaze(self):
        index = Main_window.index
        index[0] = INDEX["AKAZE"]

    def command_orb(self):
        index = Main_window.index
        index[0] = INDEX["ORB"]

    def command_desactive(self):
        index = Main_window.index
        index[0] = 0



    def switch_to_calibration(self):

        """ Function to switch activation from False to True or vice-versa when
            to push on the button"""

        self.active_calibration = not self.active_calibration

    def wipe_off(self, ident):

        """Function to make the camera calibration"""

        root = Main_window.root

        calibrate_camera = Calibration_of_camera()

        ret, frame = self.get_frame()
        ischessBoart = [0]
        self.text_calibration = "IL est nécessaire que vous presentiez un chessBoart pour faire la calibration."
        self.counter_to_calibration = self.counter_to_calibration + 1

        if self.counter_to_calibration > 30:
            ischessBoart[0] = 1

        if ischessBoart[0] == 1:

            if self.counter_to_calibration < 160 and ret:
                self.text_calibration = "Une série de 100 photos de la chessBoart vont être prise, déplacer là de temps en temps."
                if self.counter_to_calibration > 60:
                    self.canvas.delete(ident)
                    self.text_calibration = str(self.counter_to_calibration - 60) + "s"
                    cv2.imwrite(
                        "./images_to_calibration/" + "image_calib" + str(self.counter_to_calibration - 60) + ".jpg",
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                self.canvas.delete(ident)

                calibrate_camera.show_coners()
                calibrate_camera.make_calibration()

                print("-----------------------------------------------------")
                print("Fin de la calibration, relancer l'algorithme.")
                print("-----------------------------------------------------")

                root.destroy()
