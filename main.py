import cv2
import streamlit as st
import numpy as np
import av
# import time
import autopy
# import mediapipe as mp
from PIL import Image
from streamlit_webrtc import (
    # AudioProcessorBase,
    ClientSettings,
    # VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
    VideoTransformerBase
)
import HandTrackingModule as htm


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        # "audio": True,
    },
)


def main():

    selected_box = st.sidebar.selectbox(
    'Pick Something Fun',
    ('Welcome','AI Virtual Mouse')
    )
    
    if selected_box == 'Welcome':
        welcome()
    if selected_box == "AI Virtual Mouse":
        vir_mouse()

def welcome():
    st.title("AI VIRTUAL MOUSE")
    "Wanna know how AI changes everything"
    "Go to the left sidebar to explore it"


class VirtualMouse(VideoTransformerBase):
    # global smoothening, wCam, hCam, frameR
    # wCam, hCam = 640,480
    # frameR = 100
    # smoothening = 7
    # global pTime
    # pTime=0
    def __init__(self):
        self.threshold1 = 100
        self.threshold2 = 200
        self.i = 0
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # pTime = 0
        wCam, hCam = 640,480
        frameR = 100
        smoothening = 7
        plocX, plocY = 0,0
        clocX, clocY = 0,0
        detector = htm.handDetector(maxHands=1)
        wScr, hScr = autopy.screen.size()
        while True:
            img = frame.to_ndarray(format="bgr24")
            img = detector.findHands(img)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lmList, bbox = detector.findPosition(img)
            if len(lmList)!=0:
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]
                fingers = detector.fingersUp()
                cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255,0,255), 2)
                if fingers[1] == 1 and fingers[2] == 0:

                    # Convert coordinates
                    x3 = np.interp(x1, (frameR, wCam-frameR), (0,wScr))
                    y3 = np.interp(y1, (frameR, hCam-frameR), (0,hScr))

                    # Smoothen values
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening

                    # Move mouse
                    autopy.mouse.move(wScr-clocX, clocY)
                    cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
                    plocX, plocY = clocX, clocY

                # Both Index and middle fingers are up
                if fingers[1] == 1 and fingers[2] == 1:

                    # Find distance between fingers
                    length, img, lineInfo = detector.findDistance(8,12,img)
                    # print(length)

                    # Click mouse if distance short
                    if length < 40:
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0,255,0), cv2.FILLED)
                        autopy.mouse.click()

            return av.VideoFrame.from_ndarray(img,format="bgr24")
     
           


def vir_mouse():
    st.title("Click START to enjoy")
    webrtc_streamer(key="mouse", 
                    video_transformer_factory=VirtualMouse,
                    mode=WebRtcMode.SENDRECV,
                    client_settings=WEBRTC_CLIENT_SETTINGS)


if __name__ == "__main__":
    main()
