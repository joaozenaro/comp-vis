# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20


def fingerCounter(points):
    fingerCount = 0
    fingerTips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

    if points[THUMB_TIP][0] > points[THUMB_TIP-2][0]:
        fingerCount += 1

    for tip in fingerTips:
        if points[tip][1] < points[tip-2][1]:
            fingerCount += 1

    print(fingerCount)

    return fingerCount


video = cv2.VideoCapture(0, cv2.CAP_V4L2)

video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video.set(cv2.CAP_PROP_FPS, 30)

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)

mpDraw = mp.solutions.drawing_utils

cv2.namedWindow('dedos', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dedos', 1024, 768)

while True:
    _, img = video.read()
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    results = Hand.process(imgRGB)
    handPoints = results.multi_hand_landmarks

    if handPoints:
        for point in handPoints:
            # print(point)
            mpDraw.draw_landmarks(img, point, hand.HAND_CONNECTIONS)

            points = []
            for id, coord in enumerate(point.landmark):
                cx, cy = int(coord.x * w), int(coord.y * h)
                coordinate = (cx, cy)

                cv2.putText(img,
                            str(id),
                            coordinate,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2)

                points.append(coordinate)

        if point:
            nbrFingers = fingerCounter(points)
            cv2.putText(img, str(nbrFingers), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)

    cv2.imshow('dedos', img)

    ch = cv2.waitKey(1)
    if ch == 27:
        break

cv2.destroyAllWindows()
