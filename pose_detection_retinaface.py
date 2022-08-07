"""
October 03, 2020
Author: Asif Khan
"""
import tensorflow as tf
import numpy as np
import cv2

# MTCNN face detector
#from mtcnn.mtcnn import MTCNN
#detector = MTCNN()

# RetinaFace face detector
detector_model = tf.saved_model.load('./tf_retinaface_mbv2/')

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
    
    if w2h_ratio < 0.7 or w2h_ratio > 0.9:
        #cv2.putText(frame, "width/height: {0:.2f}".format(w2h_ratio), (10,40), font, font_size, blue, 1)
        cv2.putText(frame, "Face: Narrow", (10, 140), font, font_size, red, 1)
    if eye2box_ratio > 1.5 or eye2box_ratio < 0.88:
        cv2.putText(frame, "Face: not in center of the bounding box", (10, 140), font, font_size, blue, 1)

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

def face_detector(image, image_shape_max=640, score_min=None, pixel_min=None, pixel_max=None, Ain_min=None):
    '''
    Performs face detection using retinaface method with speed boost and initial quality checks based on whole image size
    
    Parameters
    ----------
    image : uint8
        image for face detection.
    image_shape_max : int, optional
        maximum size (in pixels) of image. The default is None.
    score_min : float, optional
        minimum detection score (0 to 1). The default is None.
    pixel_min : int, optional
        mininmum face size based on heigth of bounding box. The default is None.
    pixel_max : int, optional
        maximum face size based on heigth of bounding box. The default is None.
    Ain_min : float, optional
        minimum area of face in bounding box. The default is None.

    Returns
    -------
    float array
        landmarks.
    float array
        bounding boxes.
    flaot array
        detection scores.
    float array
        face area in bounding box.

    '''

    image_shape = image.shape[:2]
    
    # perform image resize for faster detection    
    if image_shape_max:
        scale_factor = max([1, max(image_shape)/image_shape_max])
    else:
        scale_factor = 1
        
    if scale_factor > 1:        
        scaled_image = cv2.resize(image, (0, 0), fx=1/scale_factor, fy=1/scale_factor)
        bbs_all, points_all = retinaface(scaled_image)
        bbs_all[:,:4]*=scale_factor
        points_all*=scale_factor
    else:
        bbs_all, points_all = retinaface(image)
    
    bbs=bbs_all.copy()
    points=points_all.copy()
    
    # check detection score
    if score_min:
        mask=np.array(bbs[:,4]>score_min)
        bbs=bbs[mask]
        points=points[mask]
        if len(bbs)==0:
            return [],[],[],[]           

    # check pixel height
    if pixel_min: 
        pixel=bbs[:,3]-bbs[:,1]
        mask=np.array(pixel>pixel_min)
        bbs=bbs[mask]
        points=points[mask]
        if len(bbs)==0:
            return [],[],[],[]           

    if pixel_max: 
        pixel=bbs[:,3]-bbs[:,1]
        mask=np.array(pixel<pixel_max)
        bbs=bbs[mask]
        points=points[mask]
        if len(bbs)==0:
            return [],[],[],[]           

    # check face area in bounding box
    Ains = []
    for bb in bbs:
        Win=min(image_shape[1],bb[2])-max(0,bb[0])
        Hin=min(image_shape[0],bb[3])-max(0,bb[1])
        Abb=(bb[2]-bb[0])*(bb[3]-bb[1])
        Ains.append(Win*Hin/Abb*100 if Abb!=0 else 0)
    Ains = np.array(Ains)

    if Ain_min:
        mask=np.array(Ains>=Ain_min)
        bbs=bbs[mask]
        points=points[mask]
        Ains=Ains[mask]
        if len(bbs)==0:
            return [],[],[],[]           
    
    scores = bbs[:,-1]
    bbs = bbs[:, :4]
    
    return points, bbs, scores, Ains

def retinaface(image):

    height = image.shape[0]
    width = image.shape[1]
    
    image_pad, pad_params = pad_input_image(image)    
    image_pad = tf.convert_to_tensor(image_pad[np.newaxis, ...])
    image_pad = tf.cast(image_pad, tf.float32)  
   
    outputs = detector_model(image_pad).numpy()

    outputs = recover_pad_output(outputs, pad_params)
    Nfaces = len(outputs)
    
    bbs = np.zeros((Nfaces,5))
    lms = np.zeros((Nfaces,10))
    
    bbs[:,[0,2]] = outputs[:,[0,2]]*width
    bbs[:,[1,3]] = outputs[:,[1,3]]*height
    bbs[:,4] = outputs[:,-1]
    
    lms[:,0:5] = outputs[:,[4,6,8,10,12]]*width
    lms[:,5:10] = outputs[:,[5,7,9,11,13]]*height
    
    return bbs, lms

def pad_input_image(img, max_steps=32):
    """pad image to suitable shape"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT, value=padd_val.tolist())
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return img, pad_params

def recover_pad_output(outputs, pad_params):
    """recover the padded output effect"""
    img_h, img_w, img_pad_h, img_pad_w = pad_params
    recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
        [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
    outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

    return outputs

#============================================================================
# FONT SETTING (font style, size and color)
font = cv2.FONT_HERSHEY_COMPLEX # Text in video
font_size = 0.4
blue = (0, 0, 255)
green = (0,128,0)
red = (255, 0, 0)

# RECORDING SETTING (Recordings on/off)
image_save = False# save face image
video_save = True# save video
fps = 10.
video_format = cv2.VideoWriter_fourcc('M','J','P','G')
if video_save:
    video_file = 'video_out.avi'
    video_out = cv2.VideoWriter(video_file, video_format, fps, (640,480))

# CAMERA SETTING (video capture initialization)
camera = 0#0: internal, 1: external
cap = cv2.VideoCapture(camera)

res_actual = np.zeros((1,2), dtype=int)# actual resolution of the camera
res_actual[0,0] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
res_actual[0,1] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print("\ncamera resolution: {}".format(res_actual))

# process each frame from camera
while (True): 
    
    ret, frame = cap.read()# read frame from camera
    if not (ret):
        print("Error: can't read from camera.")
        break
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# convert to rgb
    image_rgb = cv2.flip(image_rgb, 1)# flip for user friendliness
    
    # face detection
    try:
        pointss_all, bbs_all, scores_all, area_in = face_detector(image_rgb, image_shape_max=640, score_min=0.95,
                                                      pixel_min=20, pixel_max=1000, Ain_min=90)
    except:
        print("Error: face detector error.")
        break
    
    # if at least one face is detected
    if len(bbs_all) > 0:
        
        bbs_all = np.insert(bbs_all,bbs_all.shape[1], scores_all, axis=1)
        pointss_all = np.transpose(pointss_all)
        bbs = bbs_all.copy()
        lmarks = pointss_all.copy()
    
        # process only one face (center ?) if multiple faces detected  
        bb, lmarks_5 = one_face(image_rgb, bbs, lmarks)
        draw_landmarks(image_rgb, bb, lmarks_5)# draw landmarks and bbox   
        
        cv2.putText(image_rgb, "Face Pose", (10, 40), font, 0.8, blue, 2) 
        cv2.putText(image_rgb, "Method 1", (10, 60), font, font_size, blue, 2) 
        cv2.putText(image_rgb, "Roll: {0:.2f} (-50 to +50)".format(find_roll(lmarks_5)), (10, 80), font, font_size, blue, 1)  
        cv2.putText(image_rgb, "Yaw: {0:.2f} (-100 to +100)".format(find_yaw(lmarks_5)), (10, 100), font, font_size, blue, 1)
        cv2.putText(image_rgb, "Pitch: {0:.2f} (0 to 4)".format(find_pitch(lmarks_5)), (10, 120), font, font_size, blue, 1)
        
        angle, Xfrontal, Yfrontal = find_pose(lmarks_5)
        cv2.putText(image_rgb, "Method 2", (10, 160), font, font_size, blue, 2)
        cv2.putText(image_rgb, "Roll: {0:.2f} degrees".format(angle), (10,180), font, font_size, blue, 1)
        cv2.putText(image_rgb, "Yaw: {0:.2f} degrees".format(Xfrontal), (10,200), font, font_size, blue, 1)
        cv2.putText(image_rgb, "Pitch: {0:.2f} degrees".format(Yfrontal), (10,220), font, font_size, blue, 1)
        
        # smile detection
        smile_ratio = find_smile(lmarks_5) 
        if smile_ratio > 0.85:
            cv2.putText(image_rgb, "Smile: Yes", (10,260), font, font_size, green, 2)
        else:
            cv2.putText(image_rgb, "Smile: No", (10,260), font, font_size, green, 2)
            
    else:
        cv2.putText(image_rgb, 'no face detected', (10, 20), font, font_size, blue, 2)
    
    cv2.putText(image_rgb, "Press Q to quit.", (10, 460), font, font_size, blue, 1)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)            
    cv2.imshow('Pose Detection - Retina Face', image_bgr)    
    
    # write original frames to disk
    if video_save:
        frame = cv2.resize(frame, (640, 480))
        video_out.write(frame)        
        
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()

if video_save:
    video_out.release()

cv2.destroyAllWindows()


