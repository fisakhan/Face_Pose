import cv2
import numpy as np

class SCRFD():
    def __init__(self, onnxmodel, confThreshold=0.5, nmsThreshold=0.5):
        self.inpWidth = 640
        self.inpHeight = 640
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.net = cv2.dnn.readNet(onnxmodel)
        #self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.keep_ratio = True
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
    def resize_image(self, srcimg):
        padh, padw, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, self.inpWidth - neww - padw, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale) + 1, self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, padh, padw
    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    def distance2kps(self, points, distance, max_shape=None):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)
    def detect(self, srcimg):
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, (self.inpWidth, self.inpHeight), (127.5, 127.5, 127.5), swapRB=True)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # inference output
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outs[idx][0]
            bbox_preds = outs[idx + self.fmc][0]  * stride
            kps_preds = outs[idx + self.fmc * 2][0] * stride
            height = blob.shape[2] // stride
            width = blob.shape[3] // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            pos_inds = np.where(scores >= self.confThreshold)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list).ravel()
        # bboxes = np.vstack(bboxes_list) / det_scale
        # kpss = np.vstack(kpss_list) / det_scale
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)
        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow
        bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
        bboxes[:, 2] = bboxes[:, 2] * ratiow
        bboxes[:, 3] = bboxes[:, 3] * ratioh
        kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
        kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.confThreshold, self.nmsThreshold)
        
        '''
        for i in indices:
            #i = i[0]
            xmin, ymin, xamx, ymax = int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 0] + bboxes[i, 2]), int(bboxes[i, 1] + bboxes[i, 3])
            cv2.rectangle(srcimg, (xmin, ymin), (xamx, ymax), (0, 0, 255), thickness=2)
            for j in range(5):
                cv2.circle(srcimg, (int(kpss[i, j, 0]), int(kpss[i, j, 1])), 1, (0,255,0), thickness=-1)
            cv2.putText(srcimg, str(round(scores[i], 3)), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        '''
        return bboxes[indices], kpss[indices], scores[indices]#indices#srcimg

def visualize(image, boxes, lmarks, scores, fps=0):
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 0] + boxes[i, 2]), int(boxes[i, 1] + boxes[i, 3])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
        for j in range(5):
            cv2.circle(image, (int(lmarks[i, j, 0]), int(lmarks[i, j, 1])), 1, (0,255,0), thickness=-1)
        cv2.putText(frame, str(round(scores[i], 3)), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        cv2.putText(image, f"FPS={int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
    return image        

def are_coordinates_in_frame(frame, box, pts):
    """
    Parameters
    ----------
    frame : uint8
        RGB image (numpy array).
    bbs : float64
        coordinates of bounding box.
    points : flaot32
        coordinates of landmarks.

    Returns
    -------
    boolean
    """
    
    height, width = frame.shape[:2]
    
    if np.any(box <= 0) or np.any(box >= height) or np.any(box >= width):
        return False
    if np.any(pts <= 0) or np.any(pts >= height) or np.any(pts >= width):
        return False
    
    return True

def find_pose(points):
    """
    Parameters
    ----------
    points : float32, Size = (5,2)
        coordinates of landmarks for the selected faces.
    Returns
    -------
    float32, float32, float32
    """
    LMx = points[:,0]#points[0:5]# horizontal coordinates of landmarks
    LMy = points[:,1]#[5:10]# vertical coordinates of landmarks
    
    dPx_eyes = max((LMx[1] - LMx[0]), 1)
    dPy_eyes = (LMy[1] - LMy[0])
    angle = np.arctan(dPy_eyes / dPx_eyes) # angle for rotation based on slope
    
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

    
# load scrfd face detector model
onnxmodel = 'models/scrfd_500m_kps.onnx'
confThreshold = 0.5
nmsThreshold = 0.5
mynet = SCRFD(onnxmodel, confThreshold=confThreshold, nmsThreshold=nmsThreshold)

deviceId = 0# select camera
cap = cv2.VideoCapture(deviceId)

tm = cv2.TickMeter()

while cv2.waitKey(1) < 0:
    
    hasFrame, frame = cap.read()
    if not hasFrame:
        print('No frames captured!')
        break
    frame = cv2.flip(frame, 1)

    # Inference
    tm.start()# for calculating FPS
    bboxes, lmarks, scores = mynet.detect(frame)# face detection
    tm.stop()
    
    # process if at least one face detected
    if bboxes.shape[0] > 0 or lmarks.shape[0] > 0:
        
        # Draw results on the input image
        frame = visualize(frame, bboxes, lmarks, scores, fps=tm.getFPS())
        
        # Check if all coordinates of the highest score face in the frame
        if are_coordinates_in_frame(frame, bboxes[0], lmarks[0]):
            
            roll, yaw, pitch = find_pose(lmarks[0])
            
            # visualize pose
            lmarks = lmarks.astype(int)
            start_point = (lmarks[0][2][0], lmarks[0][2][1])
            end_point = (lmarks[0][2][0]-int(yaw), lmarks[0][2][1]-int(pitch))
            
            cv2.arrowedLine(frame, start_point, end_point, (255,0,0), 2)
            bn = "\n"
            cv2.putText(frame, f"roll: {int(roll)} -- yaw: {int(yaw)} -- pitch: {int(pitch)}", 
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
    
    # Visualize results in a new Window
    cv2.imshow('Face Pose', frame)
    #cv2.waitKey(0)

    tm.reset()


cv2.destroyAllWindows()
cap.release()
