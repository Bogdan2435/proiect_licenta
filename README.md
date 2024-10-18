# Autonomous Driving Robot

This project is related to the development of an autonomous driving robot, which was part of my bachelor's thesis. The robot was developed by starting with the [open-source platform ORP](https://openroboticplatform.com/) and designing new components or customizing some of the pats. For the development of the robot, I used a Raspberry Pi 5 as the main controller and a GoPro camera connected to it for the vision capabilities of the robot.

<img src="./images/robot_colectare_date.jpg" alt="Robot Image" width="475"> 

The robot can navigate autonomously by predicting the steering angle for each frame captured by the camera, and then computing the actual motor wheel power to follow the desired trajectory. It can detect traffic signs and lights and react accordingly. For example, when a stop sign is detected, the robot will stop in front of the sign and wait for 2 seconds or when the red light is detected, it will stop until the red light turns green. The model used for the prediction of the steering angle is a convolutional neural network, Resnet18, which was fine-tuned on a dataset of images captured with the robot and manually annotated. Traffic signs and lights detection is performed using the pretrained YOLOv8n model fine-tuned on a dataset of images captured with the robot and manually annotated.

A demo video part of an entire playlist of videos related to the robot and the development process can be found below:

[![Youtube playlist](https://img.youtube.com/vi/oqh-v6hqUFc/0.jpg)](https://www.youtube.com/watch?v=oqh-v6hqUFc&list=PLAl-NcXT4JjrTmu2yR6H8CIJtusMD7gc9)

To develop the autonomous robot, I had to prepare the environment for the robot, which consisted of strips of paper (as the lane markings), traffic signs, and traffic lights that were 3D modeled and printed. Each traffic light was equipped with two LEDs, an Arduino Nano board for control, and powered from a recycled disposable vape battery.

<img src="./images/traffic_signs-light.png" alt="Traffic signs and lights Image" width="500"> 
