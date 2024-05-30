import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandDrawingTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.imgcanvas = None
        self.xp, self.yp = 0, 0
        self.prev_time = 0
        self.curr_time = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if self.imgcanvas is None:
            self.imgcanvas = np.zeros_like(img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

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
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = point_index
                    cv2.circle(img, point_index, 10, (0, 0, 255), cv2.FILLED)
                    cv2.line(self.imgcanvas, (self.xp, self.yp), point_index, (255, 0, 0), 10)
                    self.xp, self.yp = point_index

                if fingers[0] == 1 and fingers[1] == 1:
                    self.xp, self.yp = 0, 0
                    cv2.rectangle(img, point_index, point_middle, (255, 255, 255), cv2.FILLED)
                    cv2.rectangle(self.imgcanvas, point_index, point_middle, (0, 0, 0), 20, cv2.FILLED)

                if fingers[0] == 0 and fingers[1] == 0:
                    self.xp, self.yp = 0, 0

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        imgGray = cv2.cvtColor(self.imgcanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, self.imgcanvas)

        self.curr_time = time.time()
        fps = 1 / (self.curr_time - self.prev_time)
        self.prev_time = self.curr_time

        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img

st.title("Virtual Whiteboard")
st.write("Project by Jaskirat Singh Sudan")

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
webrtc_streamer(key="example", rtc_configuration=rtc_config, video_transformer_factory=HandDrawingTransformer)