
from tkinter import *

root = Tk()
cnv = Canvas(root, width=400, height=400)
cnv.pack()



old=None

def rayon(r):
    global old
    r=int(r)
    cnv.delete(old)
    old=cnv.create_oval(200-r,200-r,200+r, 200+r)


curseur = Scale(root, orient = "horizontal",

command=rayon, from_=0, to=200)
curseur.pack()



root.mainloop()





"""import cv2 as cv
img = cv.imread("./images/hamed9_.jpg")

viewimage = img[10:565, 10:400]
print(img.shape)
print(viewimage.shape)
cv.imwrite("/home/hamed/My_projects/Machine_Learning/Computer_vision/features_matching/images/images_coffres/you.jpg", viewimage)"""

"""def of_circle(s):
    def draw_circle(event,x,y,flags,param):
        if event == cv.EVENT_LBUTTONDBLCLK:
            x1 = x-s
            y1 = y-s
            x2 = x+s
            y2 = y+s
            #crop = img[50:180, 100:300]
            #cv.imshow("Homography", crop)
            cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)

            #cv.circle(img,(x,y),300,(255,0,0),-1)
    return draw_circle




# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)+255
def back(*args):
    pass



cv.namedWindow('image',cv.WND_PROP_FULLSCREEN)
cv.setMouseCallback('image', of_circle(100))
while(1):
    cv.imshow('image', img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()

# Import required Libraries"""
"""from tkinter import *
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


