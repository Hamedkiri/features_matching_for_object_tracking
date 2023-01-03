# Features-matching to track object

## Working principle of the features-matching algorithm

When you observe an object, you can recognize it in different contexts and different photos where the object is present. This means that in different images,
different photos of an object some things don't change otherwise you would not be able to recognize it. The goal of the **feature matching** algorithm is to find the aspects,
the pixels of the image which do not change, describe them and compare them with other images which show the same object. The name of these pixels is **keypoints**.


Thereby the **features-matching** algorithms proceed in three steps : the **detection**, the **description** and the **matching** of **keypoints**.

<ins>**important note**</ins>

As we said, the feature matching algorithm can detect image pixels that do not change, precisely those pixels that are robust to photometric,
scale, rotation, ... changes. But it may be that the number of pixels found is not sufficient to reconstruct the image. therefore we will look for at 
least four pixels, three of which are not aligned, which correspond between the two images. Then from these four pixels we are going to deduce 
the homography which makes it possible to pass from one image to another. If there is a homography we can say that the two images correspond. 
**Thus, from the correspondence between pixels we pass to the correspondence between images**.

## The packages needed to use the algorithm

* Install : **Python 3.6.13** (It is recommended to create a virtual environment. If you are used to using Pycharm or Anaconda it will be easy.)
* Install requirement.txt file : **pip3 install -r requirements.txt**
  
 


