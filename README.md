study Bro Guard

A computer vision project that detects when I am using my phone while I am supposed to be studying.

The program uses a webcam to detect a smartphone and track hand landmarks. When the phone is detected in my hand for a certain amount of time, it triggers a warning and eventually opens the KFC careers page.

How It Works
Phone detected
      +
Hand landmarks detected
      ↓
Phone is detected in the hand
      ↓
Timer starts
      ↓
First warning
      ↓
After the set time
      ↓
Final warning
      ↓
KFC careers page opens

The phone is detected using YOLO, while MediaPipe Hand Landmarker is used to detect 21 landmarks on the hand. The program checks how many hand landmarks are inside or close to the detected phone area to determine whether the phone is being held.

Technologies Used
Python
OpenCV
YOLO (Ultralytics)
MediaPipe
Installation

Clone the repository:

git clone https://github.com/inksrini/study-bro.git
cd study-bro

Install the required packages:

python -m pip install -r requirements.txt

The project requires OpenCV, Ultralytics, and MediaPipe.

Run the program:

python main.py

On the first run, the required hand landmark model is downloaded automatically.

Configuration

The detection sensitivity and warning timings can be changed in main.py.

PHONE_CONFIDENCE = 0.30
HAND_MARGIN = 35
MIN_HAND_LANDMARKS_NEAR_PHONE = 3

FIRST_WARNING_SECONDS = 3
KFC_SECONDS = 8

HAND_MARGIN controls how close a hand landmark needs to be to the phone, while MIN_HAND_LANDMARKS_NEAR_PHONE controls how many landmarks are required before the program considers the phone to be in the hand.

Limitations

The system works with a 2D webcam image, so fingers can be hidden behind the phone or partially blocked from view. Because of this, the detection is not guaranteed to be accurate in every situation.

Why I Built It

I wanted to learn more about computer vision by building something practical around a problem I actually have: getting distracted by my phone while studying.

The KFC careers page is there as a final punishment because apparently normal motivation was not enough.
