# Face_Pose
<b>Find pose (Yaw, Roll, Pitch) of a face in live video from webcam.</b><br>
Using two different face detectors: mtcnn and retinaface.<br>

Detects faces in a live video from webcam and selects only one face for further processing.<br> 
Roll: -x to x (0 is frontal, positive is clock-wise, negative is anti-clock-wise)<br>
Yaw:  -x to x (0 is frontal, positive is looking right, negative is looking left)<br>
Pitch: 0 to 4 (0 is looking upward, 1 is looking straight, >1 is looking downward)<br><br>
Roll, Yaw and Pitch are in pixels and provide an estimate of where the face is looking at. In case of mtcnn Roll is (-50 to 50), Yaw is (-100 to 100). Pitch is 0 to 4 because you can divide the distance between eyes and lips into 4 units where one unit is between lips to nose-tip and 3 units between nose-tip to eyes.<br>

Xfrontal and Yfrontal provide pose (Yaw and Pitch only) in terms of angles (in degrees) along X and Y axis, respectively. These values are obtained after compensating the roll (aligning both the eyes horizontally).

<b>Tested on:</b><br>
Ubuntu-18.04<br>
numpy-1.17.0 to 1.19.1<br>
tensorflow-2.0.0 to 2.3.0<br>
opencv-python-4.0.0.21 to 4.4.0.42<br>

<b>Acknowledgement:</b><br>
Face detection using mtcnn https://github.com/ipazc/mtcnn.<br>
Face detection using retinaface https://github.com/peteryuX/retinaface-tf2
