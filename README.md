# AI-Based Virtual Mouse Using Hand Gesture Recognition

A real-time virtual mouse system that allows users to control the mouse cursor and perform click actions using hand gestures captured through a webcam.
The project uses computer vision and geometric analysis of hand landmarks to enable touch-free human–computer interaction.

## Key Features

Real-time hand tracking using MediaPipe

Cursor movement using index finger position

Gesture-based left click, right click, and double click

Screenshot capture using a fist gesture

Smooth cursor movement with jitter reduction

Cooldown logic to prevent accidental repeated actions

## Technologies Used

Python

OpenCV

MediaPipe

NumPy

PyAutoGUI

Pynput

## How It Works (High Level)

Webcam captures live video feed

MediaPipe detects 21 hand landmarks per frame

Finger bending is determined using angle calculations

Distance between thumb and index finger is used for gesture separation

Detected gestures are mapped to mouse actions in real time
