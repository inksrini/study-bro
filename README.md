# Study Bro Guard

## A Computer Vision Productivity Guard

A computer vision project that detects when I am using my phone while I am supposed to be studying.

## How It Works

Phone detected  
+  
Hand landmarks detected  
-> 
Phone is detected in the hand  
->
Timer starts  
-> 
First warning  
->  
After the set time  
->
Final warning  
-> 
KFC careers page opens

The phone is detected using YOLO, while MediaPipe Hand Landmarker is used to detect 21 landmarks on the hand. The program checks how many hand landmarks are inside or close to the detected phone area to determine whether the phone is being held.

## Technologies Used

- Python
- OpenCV
- YOLO (Ultralytics)
- MediaPipe

## Installation

### Clone the repository

```bash
git clone https://github.com/inksrini/study-bro.git
cd study-bro
