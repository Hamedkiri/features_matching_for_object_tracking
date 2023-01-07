import cv2
import glob
from PIL import Image, ImageTk

register_video = []


def write_video(file_path, frames, fps, width, height):
    """
    Writes frames to an mp4 video file
    :param height:
    :param width:
    :param side:
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
link = glob.glob("/home/hamed/My_projects/Machine_Learning_Computer_Vision/Computer_vision/brouillon/doss/*.jpg")
images=[]
for a in link:
    images.append(cv2.imread(a))
#print(images)

write_video(file_path="./videos/test.avi", frames=images, fps=2.0, width=images[0].shape[0], height=images[0].shape[1])

