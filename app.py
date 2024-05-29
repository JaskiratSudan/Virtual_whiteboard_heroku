from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import time

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def generate_frames():
    resolution = (640, 480)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(3, resolution[0])
    cap.set(4, resolution[1])

    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
    
    imgcanvas = np.zeros((resolution[1], resolution[0], 3), np.uint8)
    xp, yp = 0, 0

    prev_time = time.time()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lmlist = [(int(point.x * img.shape[1]), int(point.y * img.shape[0])) for point in hand_landmarks.landmark]
                point_index = lmlist[8]
                point_middle = lmlist[12]

                fingers = []
                for i in [8, 12, 16, 20]:
                    if lmlist[i][1] < lmlist[i - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                if fingers[0] == 1 and fingers[1] == 0:
                    if xp == 0 and yp == 0:
                        xp, yp = point_index
                    cv2.circle(img, point_index, 10, (0, 0, 255), cv2.FILLED)
                    cv2.line(imgcanvas, (xp, yp), point_index, (255, 0, 0), 10)
                    xp, yp = point_index

                if fingers[0] == 1 and fingers[1] == 1:
                    xp, yp = 0, 0
                    cv2.rectangle(img, point_index, point_middle, (255, 255, 255), cv2.FILLED)
                    cv2.rectangle(imgcanvas, point_index, point_middle, (0, 0, 0), 20, cv2.FILLED)

                if fingers[0] == 0 and fingers[1] == 0:
                    xp, yp = 0, 0

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        imgGray = cv2.cvtColor(imgcanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgcanvas)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
