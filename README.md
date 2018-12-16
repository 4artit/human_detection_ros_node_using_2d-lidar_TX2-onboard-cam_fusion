# ROS Package for human_detection_with_sensor_fusion

## About
- Cloned from github/gustav/realtime_object_detection [Link Base Repository](https://github.com/GustavZ/realtime_object_detection)
- This is for TX2 realtime camera 2Dlidar fusion but if you just want to use camera you can.
- I used TX2 on board camera. If you want to use that also, see below *Using TX2 on board cam*.
- If you want to use this, first you need to calibration your cam and lidar. About calibration, you can see in **Using 2D Lidar** section.

## Setup:
- JetPack 3.1
- Tensorflow 1.4([this repo](https://github.com/peterlee0127/tensorflow-nvJetson) provides pre-build tf wheel files for jetson tx2)
- ROS Kinetic
- Python 2.7
- Numpy
- OpenCV 3.3.1(You need cv_bridge)

## How to use:
- Change some setting in *config.sample.yml*, and save as *config.yml*.

## Getting started:
- clone this repo into your catkin workspace [catkin_ws/src/]
- build your workspace: `catkin build`
- `source devel/setup.bash`
- create copy of `config.sample.yml` named `config.yml` and change parameter according to your needs
- start ros: `roscore`
- start detection_node: `rosrun objdetection detection_node`
- witness greatness!

## Using 2D Lidar:
- I used RPLIDAR A1. So you need rpdiar ros driver.
- If you want to use lidar go to *config.sample.yml*, change *LIDAR_USE* True and save as *config.yml*.
- First you need to calibrate between 2DLIDAR and Camera.
- Get *PHI, DETLA* from fusion calibration and get  *K, C, F* from Camera Calibration.
- Paste those in constants parts in *src/realtime_object_detection/ros.py*.
- Use [this repo](https://github.com/4artit/2D-Lidar_Camera_Calibration) for getting *PHI DELTA*

## Using TX2 on board cam:
- Use this node. [TX2 ON BOARD CAMERA NODE](https://github.com/zhuhaijun753/TX2_on_board_cam)
- And you have to change *ROS_INPUT* as */bottom/camera/image_raw* when you are changing *config.sample.yml*.

## Clustering
- You can use 3 kinds of method.(KNN, DBSCAN, get Mean point form N points)
- line61 ~ line93 in ros.py has the code. So if you want to use different method, comment or uncomment here.
- Default is KNN.

## Result

* HOW TO PROJECT POINTS

  ![mid_result](./asset/result2.gif)

* **TOTAL RESULT**

![result](./asset/realtime_fusion.gif)

