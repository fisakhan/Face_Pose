"""
Author: Asif Khan
"""

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

#import matplotlib.pyplot as plt

def detect_faces(image, image_shape_max=640):
    '''
    Performs face detection using retinaface method with speed boost and initial quality checks based on whole image size
    
    Parameters
    ----------
    image : uint8
        image for face detection.
    image_shape_max : int, optional
        maximum size (in pixels) of image. The default is None.

    Returns
    -------
    float array
        bounding boxes and score.
    float array
        landmarks.

    '''

    image_shape = image.shape[:2]
    
    # perform image resize for faster detection    
    if image_shape_max:
        scale_factor = max([1, max(image_shape) / image_shape_max])
    else:
        scale_factor = 1
        
    if scale_factor > 1:        
        scaled_image = cv2.resize(image, (0, 0), fx = 1/scale_factor, fy = 1/scale_factor)
        bbs, points = detector.detect_faces(scaled_image)
        bbs[:,:4] *= scale_factor
        points *= scale_factor
    else:
        bbs, points = detector.detect_faces(image)
    
    return bbs, points

def one_face(frame, bbs, pointss):
    """
    Parameters
    ----------
    frame : TYPE
        RGB image (numpy array).
    bbs : TYPE - Array of flaot64, Size = (N, 5)
        coordinates of bounding boxes for all detected faces.
    pointss : TYPE - Array of flaot32, Size = (N, 10)
        coordinates of landmarks for all detected faces.

    Returns
    -------
    bb : TYPE - Array of float 64, Size = (5,)
        coordinates of bounding box for the selected face.
    points : TYPE
        coordinates of five landmarks for the selected face.

    """
    # select only process only one face (center ?)
    offsets = [(bbs[:,0]+bbs[:,2])/2-frame.shape[1]/2,
               (bbs[:,1]+bbs[:,3])/2-frame.shape[0]/2]
    offset_dist = np.sum(np.abs(offsets),0)
    index = np.argmin(offset_dist)
    bb = bbs[index]
    points = pointss[:,index]
    return bb, points
            
def draw_landmarks(frame, bb, points):
    '''
    Parameters
    ----------
    frame : TYPE
        RGB image
    bb : TYPE - Array of float64, Size = (5,)
        coordinates of bounding box for the selected face.
    points : TYPE - Array of float32, Size = (10,)
        coordinates of landmarks for the selected faces.

    Returns
    -------
    None.

    '''
    bb = bb.astype(int)
    points = points.astype(int)
    # draw rectangle and landmarks on face
    cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), red, 1)
    cv2.circle(frame, (points[0], points[5]), 2, blue, 2)# left eye
    cv2.circle(frame, (points[1], points[6]), 2, blue, 2)# right eye
    cv2.circle(frame, (points[2], points[7]), 2, blue, 2)# nose
    cv2.circle(frame, (points[3], points[8]), 2, blue, 2)# mouth - left
    cv2.circle(frame, (points[4], points[9]), 2, blue, 2)# mouth - right 
    
    w = int(bb[2])-int(bb[0])# width
    h = int(bb[3])-int(bb[1])# height
    w2h_ratio = w/h# width to height ratio
    eye2box_ratio = (points[0]-bb[0]) / (bb[2]-points[1])
    
    #cv2.putText(frame, "Width (pixels): {}".format(w), (10,30), font, font_size, red, 1)
    #cv2.putText(frame, "Height (pixels): {}".format(h), (10,40), font, font_size, red, 1)
    
    if eye2box_ratio > 1.5 or eye2box_ratio < 0.88:
        cv2.putText(frame, "Face: not in center of the bounding box", (10, 140), font, font_size, blue, 1)
    if w2h_ratio < 0.7 or w2h_ratio > 0.9:
        cv2.putText(frame, "Face: long and narrow", (10, 160), font, font_size, blue, 1)

def find_smile(pts):
    dx_eyes = pts[1] - pts[0]# between pupils
    dx_mout = pts[4] - pts[3]# between mouth corners
    smile_ratio = dx_mout/dx_eyes    
    return smile_ratio

def find_roll(points):
    """
    Parameters
    ----------
    points : TYPE - Array of float32, Size = (10,)
        coordinates of landmarks for the selected faces.
    Returns
    -------
    TYPE
        roll of face.

    """
    return points[6] - points[5]

def find_yaw(points):
    """
    Parameters
    ----------
    points : TYPE - Array of float32, Size = (10,)
        coordinates of landmarks for the selected faces.
    Returns
    -------
    TYPE
        yaw of face.

    """
    le2n = points[2] - points[0]
    re2n = points[1] - points[2]
    return le2n - re2n

def find_pitch(points):
    """
    Parameters
    ----------
    points : TYPE - Array of float32, Size = (10,)
        coordinates of landmarks for the selected faces.
    Returns
    -------
    Pitch
    """
    eye_y = (points[5] + points[6]) / 2
    mou_y = (points[8] + points[9]) / 2
    e2n = eye_y - points[7]
    n2m = points[7] - mou_y
    return e2n / n2m

def find_pose(points):
    """
    Parameters
    ----------
    points : TYPE - Array of float32, Size = (10,)
        coordinates of landmarks for the selected faces.
    Returns
    -------
    Angle
    Yaw
    Pitch
    TYPE
        pitch of face.

    """
    LMx = points[0:5]# horizontal coordinates of landmarks
    LMy = points[5:10]# vertical coordinates of landmarks
    
    dPx_eyes = max((LMx[1] - LMx[0]), 1)
    dPy_eyes = (LMy[1] - LMy[0])
    angle = np.arctan(dPy_eyes / dPx_eyes) # angle for rotation based on slope eyes
    
    alpha = np.cos(angle)
    beta = np.sin(angle)
    # rotated landmarks
    LMxr = (alpha * LMx + beta * LMy + (1 - alpha) * LMx[2] / 2 - beta * LMy[2] / 2) 
    LMyr = (-beta * LMx + alpha * LMy + beta * LMx[2] / 2 + (1 - alpha) * LMy[2] / 2)
    
    # average distance between eyes and mouth
    dXtot = (LMxr[1] - LMxr[0] + LMxr[4] - LMxr[3]) / 2
    dYtot = (LMyr[3] - LMyr[0] + LMyr[4] - LMyr[1]) / 2
    
    # average distance between nose and eyes
    dXnose = (LMxr[1] - LMxr[2] + LMxr[4] - LMxr[2]) / 2
    dYnose = (LMyr[3] - LMyr[2] + LMyr[4] - LMyr[2]) / 2
    
    # relative rotation 0 degree is frontal 90 degree is profile
    Xfrontal = (-90+90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0
    Yfrontal = (-90+90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0

    return angle * 180 / np.pi, Xfrontal, Yfrontal


font = cv2.FONT_HERSHEY_COMPLEX # Text in video
font_size = 0.6
blue = (0, 0, 255)
green = (0,128,0)
red = (255, 0, 0)

#print('initializing variables...')
#minsize = 20 # minimum size of face
#threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
#factor = 0.709 # scale factor

# Recordings on/off
image_save = False
video_save = False
fps = 10.
video_format=cv2.VideoWriter_fourcc('M','J','P','G')
#video_max_frame=60
#video_outs=[]

# video capture initialization
camera = 0#0: internal, 1: external
cap = cv2.VideoCapture(camera)

res_actual = np.zeros((1,2), dtype=int)# initialize resolution
res_actual[0,0]=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
res_actual[0,1]=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print("camera resolution: {}".format(res_actual))

if video_save:
    video_file = 'video_out.avi'
    video_out = cv2.VideoWriter(video_file, video_format, fps, (640, 480))

# process each frame from camera
while (True): 
    
    rets, frame = cap.read()
    if not (rets):
        print("Error: can't read from camera.")
        break
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# convert to rgb
    image_rgb = cv2.flip(image_rgb, 1)# flip for user friendliness
    
    # face detection
    try:
        #bounding_boxes, landmarks = detector.detect_faces(image_rgb)
        bounding_boxes, landmarks = detect_faces(image_rgb)
        bbs = bounding_boxes.copy()
        lmarks = landmarks.copy()
    except:
        print("Error: face detector error.")
        break
    
    # if at least one face is detected
    if len(bounding_boxes) > 0:
        
        # process only one face (center ?) if multiple faces detected 
        bb, lmarks_5 = one_face(image_rgb, bbs, lmarks)
        draw_landmarks(image_rgb, bb, lmarks_5)# draw landmarks and bbox 
        
        cv2.putText(image_rgb, "Face Pose", (10, 40), font, 0.8, blue, 2) 
        cv2.putText(image_rgb, "Method 1", (10, 60), font, font_size, blue, 2) 
        cv2.putText(image_rgb, "Roll: {0:.2f} (-50 to +50)".format(find_roll(lmarks_5)), (10, 80), font, font_size, blue, 1)  
        cv2.putText(image_rgb, "Yaw: {0:.2f} (-100 to +100)".format(find_yaw(lmarks_5)), (10, 100), font, font_size, blue, 1)
        cv2.putText(image_rgb, "Pitch: {0:.2f} (0 to 4)".format(find_pitch(lmarks_5)), (10, 120), font, font_size, blue, 1)
        
        angle, Xfrontal, Yfrontal = find_pose(lmarks_5)
        cv2.putText(image_rgb, "Method 2", (10, 180), font, font_size, blue, 2)
        cv2.putText(image_rgb, "Roll: {0:.2f} degrees".format(angle), (10,200), font, font_size, blue, 1)
        cv2.putText(image_rgb, "Yaw: {0:.2f} degrees".format(Xfrontal), (10,220), font, font_size, blue, 1)
        cv2.putText(image_rgb, "Pitch: {0:.2f} degrees".format(Yfrontal), (10,240), font, font_size, blue, 1)
        
        # smile detection
        smile_ratio = find_smile(lmarks_5) 
        if smile_ratio > 0.9:
            cv2.putText(image_rgb, "Smile: Yes", (10,280), font, font_size, blue, 2)
        else:
            cv2.putText(image_rgb, "Smile: No", (10,280), font, font_size, blue, 2)
    else:
        cv2.putText(image_rgb, 'no face detected', (10, 20), font, font_size, blue, 2)
    
    if video_save:
        frame = cv2.resize(frame, (640, 480))
        video_out.write(frame)
    
    cv2.putText(image_rgb, "Press Q to quit.", (10, 460), font, font_size, blue, 1)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow('Face Pose Detection - MTCNN', image_bgr) 
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
cap.release()

if video_save:
    video_out.release()

cv2.destroyAllWindows()


