# Real-time_head_pose_estimation
Real-time head pose estimation


This code is an implementation of head pose estimation (for a single image and webcam). 
For head pose estimation, I used the algorithm explained [here](https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/), and integrated it with dlib's facial landmark detector to make it real-time. 

Please download dlib pre-trained facial landmark detector [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). 

## Sample result for a single image
![Result](https://user-images.githubusercontent.com/38423900/76159128-9606bc80-6160-11ea-912a-78a57f307f45.png)


# Dependencies:

* Python 3.6.8

* numpy 1.18.1

* Opencv 4.2.0


# How to install dlib
## Linux
You can install dlib by:
```
pip install dlib
```

## Windows
“Pip install dlib” doesn’t work for Windows. Therefore, use the following procedure:
First download “wheel” from [here](https://pypi.org/simple/dlib). 

In the above link, you can see several versions. For example, for 64-bit Windows and Python 3.6, download the following wheel file:

dlib-19.8.1-cp36-cp36m-win_amd64.whl

Then, open Windows command prompt, go to the directory where the wheel file is saved by “cd” command. Then, install dlib using following command: 
```
pip install [wheel file name]
```

example:
```
pip install dlib-19.8.1-cp36-cp36m-win_amd64.whl
```

If you would like to install dlib in a virtual environment, activate the desired virtual environment before running the above command.

Finally, close the command prompt and open a new command prompt. Now, you can import and use dlib. 

**Note:** There is no wheel file for Python 3.7 for Windows environment. Please use Python 3.6. 


### Installation by conda install
Dlib can be installed by conda as explaniend [here](https://anaconda.org/menpo/dlib).



# Sources

https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib
https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python
