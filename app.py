from flask import Flask, render_template, Response
from imutils.video import VideoStream
import cv2
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

def generate_frames():
    resolution = (640, 480)  # Set the desired lower resolution here
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(3, resolution[0])
    cap.set(4, resolution[1])

    detector = HandDetector(detectionCon=0.8, maxHands=1)

    points = np.empty((2,0), np.int32)
    imgcanvas = np.zeros((resolution[1], resolution[0], 3), np.uint8)
    xp, yp = 0, 0

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)

        if hands:
            lmlist = hands[0]["lmList"]
            point_index = lmlist[8][0:2]
            point_middle = lmlist[12][0:2]

            fingers = detector.fingersUp(hands[0])

            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                cv2.rectangle(img, (point_index[0], point_index[1]), (point_middle[0], point_middle[1]), (255, 255, 255), cv2.FILLED)
                cv2.rectangle(imgcanvas, (point_index[0], point_index[1]), (point_middle[0], point_middle[1]), (0, 0, 0), 20, cv2.FILLED)

            if fingers[1] and not fingers[2]:
                if xp == 0 and yp == 0:
                    xp = point_index[0]
                    yp = point_index[1]
                cv2.circle(img, point_index, 10, (0, 0, 255), cv2.FILLED)
                cv2.line(imgcanvas, (xp, yp), (point_index[0], point_index[1]), (255, 0, 0), 10)

                xp, yp = point_index[0], point_index[1]

            if not fingers[1] and not fingers[2]:
                xp, yp = 0, 0

        imgGray = cv2.cvtColor(imgcanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgcanvas)

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
    app.run(debug=True)
