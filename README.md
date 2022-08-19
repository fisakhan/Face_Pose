# Face_Pose
<b>Based on output of a face detector, this repository intends to estimate the pose of a detected human face.</b><br>
This repository provides two different methods to estimate the pose (rotation angles) of a detected human face. Pose includes three parameters: Roll, Yaw, and Pitch (as shown in the following picture). The repository uses two different face detectors (MTCNN and RetinaFace) to show the effectiveness of the methods.<br>

![Alt text](./pose_demo.png?raw=true "Title")

<b>Method 1</b><br>
Roll: -x to x (0: no rotation, positive: clock-wise rotation, negative: anti-clock-wise rotation)<br>
Yaw:  -x to x (0: no rotation, positive: looking right, negative: looking left)<br>
Pitch: 0 to 4 (0: looking upward, 1: looking straight, >1: looking downward)<br><br>
Roll, Yaw and Pitch are in pixel ratios and provide an estimate of where the face is looking at. In case of mtcnn Roll is (-50 to 50), Yaw is (-100 to 100). Pitch is 0 to 4 because you can divide the distance between eyes and lips into 4 units where one unit is between lips to nose-tip and 3 units between nose-tip to eyes.<br>

<b>Method 2: as accurate as (NIST Face Image Quality Assessment)[https://pages.nist.gov/frvt/api/FRVT_ongoing_quality_sidd_api.pdf]
]</b><br>
This method provides pose parameters in angels (degrees).<br>
Roll: -90 to 90 (0: no rotation, positive: clock-wise rotation, negative: anti-clock-wise rotation)<br>
Yaw:  -90 to 90 (0: no rotation, positive: looking left, negative: looking right)<br>
Pitch: -90 to 90 (0: no rotation, positive: looking upward, negative: looking downward)<br><br>

<b>Smile detection</b><br>
In addition to pose, the repository also provide some additional features like smile detection.

<b>Demo</b><br>
Each of the two python files run a demo that estimates pose of a face in a live video stream from webcam/video file.<br><br>

Note: These methods are very efficient and do not require huge data, training or machine learning.

<b>Tested on:</b><br>
Ubuntu 18.04, Ubuntu 22.04.1 LTS<br>
numpy==1.17.0 to numpy==1.23.2<br>
tensorboard==2.0.0 to tensorboard==2.9.1<br>
opencv-python==4.0.0.21 to opencv-python==4.6.0.66<br>

<b>Acknowledgement:</b><br>
Face detection using mtcnn https://github.com/ipazc/mtcnn.<br>
Face detection using retinaface https://github.com/peteryuX/retinaface-tf2
