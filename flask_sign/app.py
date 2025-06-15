from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import mediapipe as mp
import math
from threading import Lock

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

gesture_lock = Lock()
current_gesture = ""

gesture_meanings = {
    "Okay": "It means 'Okay' or shows approval and agreement.",
    "Dislike": "It expresses dislike or disapproval of something.",
    "Victory": "It represents victory, success, or peace.",
    "Stop": "It is used to tell someone to stop or pause.",
    "Point": "It indicates pointing toward something.",
    "Love": "It expresses love, affection, or caring feelings.",
    "No": "It clearly means 'No' or a negative response.",
    "Miss You": "It conveys the feeling of missing someone."
}

class SignLanguageConverter:
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                    min_detection_confidence=0.7, min_tracking_confidence=0.5)
    
    def detect_gesture(self, image):
        global current_gesture
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            gesture = self.get_gesture(hand_landmarks)
            if gesture:
                with gesture_lock:
                    current_gesture = gesture
            else:
                with gesture_lock:
                    current_gesture = ""
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            with gesture_lock:
                current_gesture = ""
        return image

    def get_gesture(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        index_finger_tip = hand_landmarks.landmark[8]
        middle_finger_tip = hand_landmarks.landmark[12]
        ring_finger_tip = hand_landmarks.landmark[16]
        little_finger_tip = hand_landmarks.landmark[20]

        # "Okay"
        if thumb_tip.y < index_finger_tip.y < middle_finger_tip.y < ring_finger_tip.y < little_finger_tip.y:
            return "Okay"

        # "Dislike"
        elif thumb_tip.y > index_finger_tip.y > middle_finger_tip.y > ring_finger_tip.y > little_finger_tip.y:
            return "Dislike"

        # "Victory"
        elif index_finger_tip.y < middle_finger_tip.y and abs(index_finger_tip.x - middle_finger_tip.x) < 0.2:
            return "Victory"

        # "Stop"
        elif thumb_tip.x < index_finger_tip.x < middle_finger_tip.x:
            if (hand_landmarks.landmark[2].x < hand_landmarks.landmark[5].x) and \
               (hand_landmarks.landmark[3].x < hand_landmarks.landmark[5].x) and \
               (hand_landmarks.landmark[4].x < hand_landmarks.landmark[5].x):
                return "Stop"

        # "Point"
        else:
            wrist = hand_landmarks.landmark[0]
            vector = (index_finger_tip.x - wrist.x, index_finger_tip.y - wrist.y, index_finger_tip.z - wrist.z)
            vector_len = math.sqrt(sum([v**2 for v in vector]))
            if vector_len == 0:
                return None
            vector_unit = tuple(v / vector_len for v in vector)
            reference_vector = (0, 0, -1)
            dot_product = sum([vector_unit[i]*reference_vector[i] for i in range(3)])
            angle = math.acos(dot_product) * 180 / math.pi
            if 20 < angle < 80:
                return "Point"

        # "Love" (thumb and pinky extended, others folded)
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        pinky_tip = little_finger_tip
        pinky_dip = hand_landmarks.landmark[19]
        pinky_pip = hand_landmarks.landmark[18]

        thumb_extended = thumb_tip.y < thumb_ip.y < thumb_mcp.y
        pinky_extended = pinky_tip.y < pinky_dip.y < pinky_pip.y
        index_pip = hand_landmarks.landmark[6]
        middle_pip = hand_landmarks.landmark[10]
        ring_pip = hand_landmarks.landmark[14]

        other_fingers_folded = (index_finger_tip.y > index_pip.y and
                                middle_finger_tip.y > middle_pip.y and
                                ring_finger_tip.y > ring_pip.y)

        if thumb_extended and pinky_extended and other_fingers_folded:
            return "Love"

        # "No" (thumb and index finger tips very close)
        dist_thumb_index = math.sqrt(
            (thumb_tip.x - index_finger_tip.x)**2 +
            (thumb_tip.y - index_finger_tip.y)**2 +
            (thumb_tip.z - index_finger_tip.z)**2
        )
        if dist_thumb_index < 0.05:
            return "No"

        # "Miss You" (thumb & index apart horizontally, other fingers folded)
        middle_dip = hand_landmarks.landmark[11]
        ring_dip = hand_landmarks.landmark[15]
        little_dip = hand_landmarks.landmark[19]

        fingers_folded = (middle_finger_tip.y > middle_dip.y and
                          ring_finger_tip.y > ring_dip.y and
                          little_finger_tip.y > little_dip.y)

        thumb_index_apart = (thumb_tip.x + 0.1 < index_finger_tip.x)

        if fingers_folded and thumb_index_apart:
            return "Miss You"

        return None


converter = SignLanguageConverter()
cap = cv2.VideoCapture(0)  # Change to your webcam index

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        frame = converter.detect_gesture(frame)

        with gesture_lock:
            text = current_gesture
        if text:
            cv2.putText(frame, f'Gesture: {text}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # For now, just simulate form submission
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        # You can save or send this info somewhere
        return render_template('contact.html', success=True, name=name)
    return render_template('contact.html', success=False)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_gesture')
def get_gesture():
    with gesture_lock:
        gesture = current_gesture
    meaning = gesture_meanings.get(gesture, "") if gesture else ""
    return jsonify({'gesture': gesture, 'meaning': meaning})

if __name__ == '__main__':
    app.run(debug=True)
