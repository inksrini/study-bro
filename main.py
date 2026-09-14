
import cv2
import math
import time
import webbrowser
import threading
import ctypes
from pathlib import Path
from urllib.request import urlopen, Request

from ultralytics import YOLO
import mediapipe as mp


# ============================================================
# SETTINGS
# ============================================================

PHONE_CONFIDENCE = 0.30

# How far outside the phone box a hand landmark may be and
# still count as touching/holding the phone.
HAND_MARGIN = 35

# Require at least this many hand landmarks to be around/
# overlapping the phone before calling it "in hand".
MIN_HAND_LANDMARKS_NEAR_PHONE = 3

# Timing
FIRST_WARNING_SECONDS = 3
KFC_SECONDS = 8

KFC_CAREERS_URL = "https://kfcjobs.mcidirecthire.com/Vacancy/Application?parameters=q4%2F37VDQTFWXnK7wNIeCPexNHJcIS4mGMD%2BeFJdme%2Fw%3D"

FIRST_MESSAGE = (
    "HOW TF YOU GONNA PULL A 10/10 FINE SHYT AT A MNC\n"
    "WHEN YOU HAVEN'T PULLED A OFFER LETTER FROM HR 😭💀"
)

SECOND_MESSAGE = "YOU BETTER WORK AT KFC 😭"


# Official MediaPipe Hand Landmarker model.
MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

MODEL_PATH = Path(__file__).with_name(
    "hand_landmarker.task"
)


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def ensure_hand_model():
    """
    Download the official MediaPipe hand model on first run.

    This means you don't need to manually hunt for the .task file.
    """

    if MODEL_PATH.exists():
        return

    print("Downloading hand_landmarker.task...")
    print("This is a one-time download (~8 MB).")

    request = Request(
        MODEL_URL,
        headers={
            "User-Agent": "StudyBroGuard/1.0"
        }
    )

    with urlopen(request, timeout=60) as response:
        data = response.read()

    MODEL_PATH.write_bytes(data)

    print("Hand model downloaded.")


# ============================================================
# MODELS
# ============================================================

print("Loading phone detector...")
phone_model = YOLO("yolo26n.pt")

ensure_hand_model()

print("Loading hand landmark detector...")

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.45,
    min_hand_presence_confidence=0.45,
    min_tracking_confidence=0.45,
)

hand_landmarker = HandLandmarker.create_from_options(
    hand_options
)


# ============================================================
# POPUP
# ============================================================

def popup_async(title, message):
    """
    Non-blocking Windows popup.

    The webcam loop keeps running while the popup is visible.
    """

    def show():
        ctypes.windll.user32.MessageBoxW(
            0,
            message,
            title,
            0x30
        )

    threading.Thread(
        target=show,
        daemon=True
    ).start()


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def point_to_box_distance(point, box):
    """
    Minimum pixel distance between a point and a rectangle.

    If the point is inside the rectangle, distance = 0.
    """

    px, py = point
    x1, y1, x2, y2 = box

    dx = max(
        x1 - px,
        0,
        px - x2
    )

    dy = max(
        y1 - py,
        0,
        py - y2
    )

    return math.hypot(dx, dy)


def phone_hand_relationship(
    phone_box,
    hand_points
):
    """
    Decide whether a detected hand is actually interacting
    with the phone.

    We count how many of the 21 landmarks are:
        1. inside the phone box, or
        2. very close to the phone box.

    This is much stronger than using wrist/elbow distance.
    """

    count = 0

    for point in hand_points:

        distance = point_to_box_distance(
            point,
            phone_box
        )

        if distance <= HAND_MARGIN:
            count += 1

    return count


# ============================================================
# DETECTION
# ============================================================

def detect_phone_and_hands(
    frame,
    timestamp_ms
):
    """
    Returns:
        phone_in_hand
        phone_confidence
        phone_boxes
        hand_landmarks
        best_landmark_count
    """

    # -------------------------
    # PHONE DETECTION
    # -------------------------

    phone_results = phone_model(
        frame,
        conf=PHONE_CONFIDENCE,
        verbose=False
    )

    phone_boxes = []

    for result in phone_results:

        if result.boxes is None:
            continue

        for box, conf, cls in zip(
            result.boxes.xyxy,
            result.boxes.conf,
            result.boxes.cls
        ):

            if result.names[int(cls)] != "cell phone":
                continue

            x1, y1, x2, y2 = map(
                int,
                box.tolist()
            )

            phone_boxes.append({
                "box": (x1, y1, x2, y2),
                "confidence": float(conf),
            })

    # -------------------------
    # HAND LANDMARK DETECTION
    # -------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    hand_result = hand_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    hand_landmarks = []

    for landmarks in hand_result.hand_landmarks:

        points = []

        for landmark in landmarks:

            x = int(
                landmark.x
                * frame.shape[1]
            )

            y = int(
                landmark.y
                * frame.shape[0]
            )

            points.append(
                (x, y)
            )

        hand_landmarks.append(
            points
        )

    # -------------------------
    # PHONE + HAND RELATIONSHIP
    # -------------------------

    phone_in_hand = False
    best_count = 0
    best_confidence = 0.0

    for phone in phone_boxes:

        phone_box = phone["box"]

        for hand_points in hand_landmarks:

            landmark_count = (
                phone_hand_relationship(
                    phone_box,
                    hand_points
                )
            )

            best_count = max(
                best_count,
                landmark_count
            )

            if (
                landmark_count
                >= MIN_HAND_LANDMARKS_NEAR_PHONE
            ):

                phone_in_hand = True

                best_confidence = max(
                    best_confidence,
                    phone["confidence"]
                )

    return (
        phone_in_hand,
        best_confidence,
        phone_boxes,
        hand_landmarks,
        best_count,
    )


# ============================================================
# DRAWING
# ============================================================

def draw_debug(
    frame,
    phone_boxes,
    hand_landmarks,
    phone_in_hand,
    landmark_count
):

    # -------------------------
    # PHONE BOX
    # -------------------------

    for phone in phone_boxes:

        x1, y1, x2, y2 = phone["box"]

        confidence = (
            phone["confidence"]
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"PHONE {confidence:.0%}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # -------------------------
    # HAND LANDMARKS
    # -------------------------

    for hand_points in hand_landmarks:

        for x, y in hand_points:

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        # Connect wrist to finger regions visually.
        # This is just for debugging.
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

        for a, b in connections:

            if (
                a < len(hand_points)
                and b < len(hand_points)
            ):

                cv2.line(
                    frame,
                    hand_points[a],
                    hand_points[b],
                    (0, 200, 0),
                    2
                )

    # -------------------------
    # STATUS
    # -------------------------

    status = (
        "PHONE IN HAND"
        if phone_in_hand
        else
        "PHONE NOT IN HAND"
    )

    status_color = (
        (0, 0, 255)
        if phone_in_hand
        else
        (0, 200, 0)
    )

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )

    cv2.putText(
        frame,
        f"Hand points near phone: {landmark_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trigger needs: {MIN_HAND_LANDMARKS_NEAR_PHONE}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q = quit",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        hand_landmarker.close()

        return

    print("")
    print("========================================")
    print(" STUDY BRO GUARD v4")
    print(" REAL HAND LANDMARK DETECTION")
    print("========================================")
    print("")
    print("Hold your phone.")
    print("Green points = hand landmarks.")
    print("Red box = detected phone.")
    print("Q = quit.")
    print("")

    phone_start = None
    first_warning = False
    kfc_opened = False

    timestamp_ms = 0

    try:

        while True:

            ok, frame = camera.read()

            if not ok:

                print(
                    "ERROR: Could not read webcam."
                )

                break

            frame = cv2.flip(
                frame,
                1
            )

            timestamp_ms += 33

            # -------------------------
            # DETECTION
            # -------------------------

            try:

                (
                    phone_in_hand,
                    phone_conf,
                    phone_boxes,
                    hand_landmarks,
                    landmark_count,
                ) = detect_phone_and_hands(
                    frame,
                    timestamp_ms
                )

            except Exception as error:

                print(
                    "\nDETECTION ERROR:"
                )

                print(
                    repr(error)
                )

                phone_in_hand = False
                phone_boxes = []
                hand_landmarks = []
                landmark_count = 0

            # -------------------------
            # DEBUG WINDOW
            # -------------------------

            draw_debug(
                frame,
                phone_boxes,
                hand_landmarks,
                phone_in_hand,
                landmark_count
            )

            # -------------------------
            # PHONE IN HAND
            # -------------------------

            if phone_in_hand:

                if phone_start is None:

                    phone_start = (
                        time.time()
                    )

                elapsed = (
                    time.time()
                    - phone_start
                )

                # FIRST WARNING
                if (
                    not first_warning
                    and elapsed
                    >= FIRST_WARNING_SECONDS
                ):

                    first_warning = True

                    popup_async(
                        "BRO... 💀",
                        FIRST_MESSAGE
                    )

                # KFC
                if (
                    first_warning
                    and not kfc_opened
                    and elapsed
                    >= KFC_SECONDS
                ):

                    kfc_opened = True

                    webbrowser.open(
                        KFC_CAREERS_URL
                    )

                    popup_async(
                        "FINAL WARNING 😭",
                        SECOND_MESSAGE
                    )

            else:

                # Phone no longer in hand:
                # reset punishment chain.

                phone_start = None
                first_warning = False
                kfc_opened = False

            # -------------------------
            # CAMERA
            # -------------------------

            cv2.imshow(
                "Study Bro Guard - REAL Phone-in-Hand Detector",
                frame
            )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):

                break

    finally:

        camera.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()


if __name__ == "__main__":

    main()
