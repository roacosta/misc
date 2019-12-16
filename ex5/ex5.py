'''
#Codigo 1
import numpy as np
import cv2


cap = cv2.VideoCapture('teste.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (854,480))


def KLT_Optical_Flow(cap, feature_params, lk_params):
    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # calculate features to track
    old_gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, old_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    frame = cap
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        frame = cv2.circle(frame,(a,b),10,color[i].tolist(),-1)  

    return frame


if (cap.isOpened() == False):
    print("Erro abrindo o video ou arquivo")

while(cap.isOpened()):
    # Captura quadro a quadro
    ret, frame = cap.read()

    if ret == True:

        # params for ShiTomasi corner detection
        maxCorners = np.random.randint(10,100)
        qualityLevel = 0.3
        minDistance = np.random.randint(5,20)
        blockSize = np.random.randint(5,20)
        feature_params = dict( maxCorners = maxCorners,
                            qualityLevel = qualityLevel,
                            minDistance = minDistance,
                            blockSize = blockSize )

        # Parameters for lucas kanade optical flow
        win = np.random.randint(10,50)
        winSize = (win,win)
        lk_params = dict( winSize  = winSize,
                        maxLevel = 0,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        
        img = KLT_Optical_Flow(frame,feature_params,lk_params)
        #Mostrar imagem resultante
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #plt.title('maxCorners='+str(maxCorners)+' winSize='+str(winSize)+' minDistance='+str(minDistance)+' blockSize='+str(blockSize))

        cv2.imshow('Resultado', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

#Codigo 2
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(cv.samples.findFile("teste.mp4"))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next


#Codigo 3




import numpy as np
import cv2 as cv
cap = cv.VideoCapture('teste.mp4')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.7,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (200,200),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p0 = cv.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv.destroyAllWindows()
cap.release()

'''

import cv2 as cv

capture = cv.VideoCapture('teste.mp4')

#-- Informations about the video --
fps = 30
wait = int(1/fps * 1000/1)
width = int(854)
height = int(480)
#For recording
#codec = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FOURCC)
#writer=cv.CreateVideoWriter("img/output.avi", int(codec), int(fps), (width,height), 1) #Create writer with same parameters
#----------------------------------

prev_gray = cv.CreateImage((width,height), 8, 1) #Will hold the frame at t-1
gray = cv.CreateImage((width,height), 8, 1) # Will hold the current frame

prevPyr = cv.CreateImage((height / 3, width + 8), 8, cv.CV_8UC1) #Will hold the pyr frame at t-1
currPyr = cv.CreateImage((height / 3, width + 8), 8, cv.CV_8UC1) # idem at t

max_count = 500
qLevel= 0.01
minDist = 10
prev_points = [] #Points at t-1
curr_points = [] #Points at t
lines=[] #To keep all the lines overtime

while(1):

    frame = cv.QueryFrame(capture) #Take a frame of the video

    cv.CvtColor(frame, gray, cv.CV_BGR2GRAY) #Convert to gray
    output = cv.CloneImage(frame)

    prev_points = cv.GoodFeaturesToTrack(gray, None, None, max_count, qLevel, minDist) #Find points on the image

    #Calculate the movement using the previous and the current frame using the previous points
    curr_points, status, err = cv.CalcOpticalFlowPyrLK(prev_gray, gray, prevPyr, currPyr, prev_points, (10, 10), 3, (cv.CV_TERMCRIT_ITER|cv.CV_TERMCRIT_EPS,20, 0.03), 0)


    #If points status are ok and distance not negligible keep the point
    k = 0
    for i in range(len(curr_points)):
        nb =  abs( int(prev_points[i][0])-int(curr_points[i][0]) ) + abs( int(prev_points[i][1])-int(curr_points[i][1]) )
        if status[i] and  nb > 2 :
            prev_points[k] = prev_points[i]
            curr_points[k] = curr_points[i]
            k += 1

    prev_points = prev_points[:k]
    curr_points = curr_points[:k]
    #At the end only interesting points are kept

    #Draw all the previously kept lines otherwise they would be lost the next frame
    for (pt1, pt2) in lines:
        cv.Line(frame, pt1, pt2, (255,255,255))

    #Draw the lines between each points at t-1 and t
    for prevpoint, point in zip(prev_points,curr_points):
        prevpoint = (int(prevpoint[0]),int(prevpoint[1]))
        cv.Circle(frame, prevpoint, 15, 0)
        point = (int(point[0]),int(point[1]))
        cv.Circle(frame, point, 3, 255)
        cv.Line(frame, prevpoint, point, (255,255,255))
        lines.append((prevpoint,point)) #Append current lines to the lines list


    cv.Copy(gray, prev_gray) #Put the current frame prev_gray
    prev_points = curr_points

    cv.ShowImage("The Video", frame)
    #cv.WriteFrame(writer, frame)
    cv.WaitKey(wait)