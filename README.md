# Face_Pose
Find pose (Yaw, Roll, Pitch) of a face in live video from webcam.

Detects faces in a live video from webcam and selects only one face for further processing. 
Roll: -x to x (0 is frontal, positive is clock-wise, negative is anti-clock-wise)
Yaw:  -x to x (0 is frontal, positive is looking right, negative is looking left)
Pitch: 0 to 4 (0 is looking upward, 1 is looking straight, >1 is looking downward)

Acknowledgement: Face detection is done useing mtcnn from https://github.com/ipazc/mtcnn.
