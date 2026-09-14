
# Study Bro Guard v4 — Actual Hand Detection 💀

v4 changes the definition of "phone in hand".

Earlier versions used body pose:

```text
phone close to wrist/elbow
        ↓
assume phone is in hand
```

That is only a heuristic.

This version uses the **MediaPipe Hand Landmarker**, which detects
21 landmarks on each hand: wrist, thumb joints/tip, and finger
joints/tips.

The program now does:

```text
Detect phone
     +
Detect hand landmarks
     +
At least 3 hand landmarks are inside/very close
to the phone box
     ↓
PHONE IN HAND
```

## Install

You already have Ultralytics and OpenCV.

Install MediaPipe:

```bash
python -m pip install mediapipe
```

Or:

```bash
python -m pip install -r requirements.txt
```

The current MediaPipe package has Windows x86-64 wheels available,
and MediaPipe's Hand Landmarker API supports image/video/live-stream
hand landmark detection. The hand model is downloaded automatically
by this project the first time it runs.

## Run

```bash
python main.py
```

First run:
- downloads `hand_landmarker.task`
- loads the phone detector
- opens the webcam

The model comes from Google's official MediaPipe model URL.

## Debug screen

You should see:

```text
RED BOX        = phone detection
GREEN POINTS   = the 21 hand landmarks
PHONE IN HAND  = actual trigger state
```

You will also see:

```text
Hand points near phone: X
Trigger needs: 3
```

This is extremely useful for tuning.

### Example

```text
Hand points near phone: 7
Trigger needs: 3
```

=> trigger.

```text
Hand points near phone: 1
Trigger needs: 3
```

=> no trigger.

## Tuning

At the top of `main.py`:

```python
HAND_MARGIN = 35
MIN_HAND_LANDMARKS_NEAR_PHONE = 3
```

### HAND_MARGIN

Larger:

```python
HAND_MARGIN = 60
```

More forgiving.

Smaller:

```python
HAND_MARGIN = 20
```

More strict.

### MIN_HAND_LANDMARKS_NEAR_PHONE

Larger:

```python
MIN_HAND_LANDMARKS_NEAR_PHONE = 5
```

Harder to trigger.

Smaller:

```python
MIN_HAND_LANDMARKS_NEAR_PHONE = 2
```

Easier to trigger.

Do NOT immediately crank these values around randomly.
Look at the debug counter first.

## Important limitation

This is substantially better than the old wrist/elbow method, but
it still isn't mathematically guaranteed to prove a person is gripping
the phone.

The webcam sees a 2D image. Fingers can be hidden behind the phone,
and the phone can occlude landmarks.

For an even more bulletproof system, the next level would be temporal
tracking + phone/hand bounding-box geometry + finger visibility +
multiple-frame confirmation.

## Timing

```python
FIRST_WARNING_SECONDS = 3
KFC_SECONDS = 8
```

For your final system you might prefer:

```python
FIRST_WARNING_SECONDS = 8
KFC_SECONDS = 16
```
