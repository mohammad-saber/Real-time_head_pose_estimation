
# Head Pose Estimation

# Input is an image file. Facial landmark positions are found by dlib. 

import dlib 
import cv2
import numpy as np


# *****************************************************************************
# Parameters, Input Data
# *****************************************************************************

# dlib’s pre-trained facial landmark detector
predictorPath = "./shape_predictor_68_face_landmarks.dat"

# Read Image
image = cv2.imread("./Sample.png");
image_size = image.shape   # [Height, Width]

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Data required for HPE

# Camera internals
focal_length = image_size[1]   # image width
center = (image_size[1]/2, image_size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
 
print("Camera Matrix :\n {0}".format(camera_matrix))  

# 3D model points.
model_points = np.array([   (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])


# Input vector of distortion coefficients, Assuming no lens distortion
dist_coeffs = np.zeros((4,1))   


# *****************************************************************************
# Functions
# *****************************************************************************

# rectangle to bounding box
def rect_to_bb(rect):
	
    # take a bounding predicted by dlib face detector and convert it
    # to the format (x, y, w, h) as we would normally do with OpenCV
    # The "rect" object includes the (x, y)-coordinates of the detection.
        
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
     
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


# dlib face landmark detector returns a shape object containing the 68 (x, y)-coordinates 
# of the facial landmark regions. We convert this object to a NumPy array. 
# shape is a dlib object, containing 68 landmark points (shape.part)
def shape_to_np(shape, dtype="int"):
	
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
     
    # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
     
    # return the list of (x, y)-coordinates, data-type is numpy array
    return coords


# Head Pose Estimation function
def HPS(image, shape_array):
    
    #2D image points. 
    image_points = np.array([   shape_array[30],     # Nose tip
                                shape_array[8],      # Chin
                                shape_array[36],     # Left eye left corner
                                shape_array[45],     # Right eye right corne
                                shape_array[48],     # Left Mouth corner
                                shape_array[54]      # Right mouth corner
                            ], dtype="double")
     
  
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs) #, flags=SOLVEPNP_ITERATIVE)
     
    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))
     
     
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose   
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
     
    # for p in image_points:
    #     cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)  
     
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
     
    cv2.line(image, p1, p2, (255,0,0), 2)
    

# *****************************************************************************
# Face Detection , Facial Landmark Predictor , Head Pose Estimation
# *****************************************************************************

# initialize dlib's pre-trained face detector and load the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorPath)

# detect faces in the grayscale image (rects : bounding box of faces in the image)
# each face has 4 elements : left, top, right, bottom
# 2nd parameter is the number of image pyramid layers to apply when upscaling the image  
rects = detector(gray_image, 1)


# Given the (x, y)-coordinates of the faces in the image, we can now apply facial landmark detection to each of the face regions.
# loop over the detected faces 
for (i, rect) in enumerate(rects):   
    
    # determine the facial landmarks for the face region (face region is stored in the rect)
    shape = predictor(gray_image, rect)

    # convert the facial landmark (x, y)-coordinates to a NumPy array (68*2)
    shape_array = shape_to_np(shape)
    
    # Show head pose
    HPS(image, shape_array)
     
    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     
    # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
    #for (x, y) in shape_array:
        #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)   # Negative thickness means that a filled circle is to be drawn.
 

# show the output image 
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("./Result.png", image)


