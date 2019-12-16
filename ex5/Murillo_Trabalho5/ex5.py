import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



def KLT_Optical_Flow(cap, feature_params, lk_params):
    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # calculate features to track
    old_gray = cv.cvtColor(cap, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, old_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    frame = cap
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        frame = cv.circle(frame,(a,b),10,color[i].tolist(),-1)  

    return frame

for i in range(10):
    cap = cv.imread('cars.png',1)

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
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    
    img = KLT_Optical_Flow(cap,feature_params,lk_params)
    #Mostrar imagem resultante
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.title('maxCorners='+str(maxCorners)+' winSize='+str(winSize)+' minDistance='+str(minDistance)+' blockSize='+str(blockSize))

    plt.imshow(img)
    plt.show()

