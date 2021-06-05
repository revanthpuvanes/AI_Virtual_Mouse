
from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_ngrok import run_with_ngrok
import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import mediapipe

wCam, hCam = 640,480
frameR = 100
smoothening = 7


app = Flask(__name__)
run_with_ngrok(app)

  
@app.route("/")

def start():
    pTime = 0
    plocX, plocY = 0,0
    clocX, clocY = 0,0

    cap = cv2.VideoCapture(0)
    cap.set(3,wCam)
    cap.set(4,hCam)
    pTime = 0
    detector = htm.handDetector(maxHands=1)
    wScr, hScr = autopy.screen.size()
    #print(wScr,hScr)

    while True:
        # Find hand landmarks
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        # Get the tip of index and middle fingers
        if len(lmList)!=0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            #print(x1,y1,x2,y2)

            # Check which fingers are up
            fingers = detector.fingersUp()
            #print(fingers)
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255,0,255), 2)

            # Only index fingure moving mode
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
                print(length)

                # Click mouse if distance short
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0,255,0), cv2.FILLED)
                    autopy.mouse.click()


        # Frame rate
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        # Display
        cv2.imshow("Image",img)
        cv2.waitKey(1)
app.run()