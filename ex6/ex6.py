import numpy as np
import cv2
import argparse

def mousef(event, x, y, flags, params):
    global mx,my,roi_hist
    if event == cv2.EVENT_LBUTTONDOWN:
        mx = x
        my = y
        width = 100
        height = 100
        roi = frame[my: my + height, mx: mx + width]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 30.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

video = cv2.VideoCapture('video.mp4')
#video = cv2.VideoCapture(0)

#Ler primeiro frame
ret, frame = video.read()

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mousef)

mx = 290
my = 155
width = 100
height = 100
roi = frame[my: my + height, mx: mx + width]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 30.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

#while(video.isOpened()):
while(True):
    #Ler frame
    ret, frame = video.read()
    
    #Se terminar video, sair
    if ret==False:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    _, track_window = cv2.meanShift(mask, (mx, my, width, height), term_criteria)
    mx, my, w, h = track_window
    cv2.rectangle(frame, (mx, my), (mx+w,my+h), (0, 255, 0), 2)
    
    #Mostrar na tela
    #cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#Terminar
video.release()
cv2.destroyAllWindows()
