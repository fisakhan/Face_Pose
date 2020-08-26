# Face_Pose
<b>Find pose (Yaw, Roll, Pitch) of a face in live video from webcam.</b><br>

Detects faces in a live video from webcam and selects only one face for further processing.<br> 
Roll: -x to x (0 is frontal, positive is clock-wise, negative is anti-clock-wise)<br>
Yaw:  -x to x (0 is frontal, positive is looking right, negative is looking left)<br>
Pitch: 0 to 4 (0 is looking upward, 1 is looking straight, >1 is looking downward)<br>

<b>Tested on:</b><br>
Ubuntu-18.04<br>
numpy-1.17.0 to 1.19.1<br>
tensorflow-2.0.0 to 2.3.0<br>
opencv-python-4.0.0.21 to 4.4.0.42<br>

Acknowledgement: Face detection is done useing mtcnn from https://github.com/ipazc/mtcnn.
