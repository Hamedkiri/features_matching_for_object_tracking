

import cv2

"""cap = cv2.VideoCapture(0)
i=0 #frame counter
frameTime = 1 # time of each frame in ms, you can add logic to change this value.
while(cap.isOpened()):
    ret = cap.grab() #grab frame
    i=i+1 #increment counter
    if i % 1 == 0: # display only one third of the frames, you can change this parameter according to your needs
        ret, frame = cap.retrieve() #decode frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(frameTime) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()"""

def readcam(i):

    if i % 3 == 0:
        ret, frame = cap.retrieve()  # decode frame
        return frame


cap = cv2.VideoCapture(0)
i=0 #frame counter
frameTime = 1 # time of each frame in ms, you can add logic to change this value.
while(cap.isOpened()):
    i = i + 1
    frame = readcam(i)
    cv2.imshow('frame',frame)
    if cv2.waitKey(frameTime) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()





















"""import numpy as np
import cv2 as cv
# mouse callback function



def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img,(x,y),100,(255,0,0),-1)



# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)

def back(*args):
    pass

cv.namedWindow('button')
button = cv.createButton("back", back, None,cv.QT_PUSH_BUTTON,0)

cv.rectangle(button,(384,0),(510,128),(0,255,0),3)

cv.namedWindow('image',cv.WND_PROP_FULLSCREEN)
cv.setMouseCallback('image',draw_circle)
while(1):
    cv.imshow('image',img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()

# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2

# Create an instance of TKinter Window or frame
win = Tk()

# Set the size of the window
win.geometry("1000x350")

# Create a Label to capture the Video frames
label =Label(win)
label.grid(row=0, column=0)
cap= cv2.VideoCapture(0)

# Define function to show frame
def show_frames():
   # Get the latest frame and convert into Image
   cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   img = Image.fromarray(cv2image)
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   # Repeat after an interval to capture continiously
   label.after(20, show_frames)

show_frames()
win.mainloop()"""


