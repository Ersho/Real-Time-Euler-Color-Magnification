import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import Process, Pool

from utils import detectFace, Euler_Video_Magnification
level = 3

vc = cv2.VideoCapture(0)

#width = vc.get(cv2.CV_CAP_PROP_FRAME_WIDTH)
#height = vc.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)
fps = cv2.CAP_PROP_FPS

face_detector= detectFace()

# Reduce level and amplification for motion
EVM = Euler_Video_Magnification(level = 4, 
                                amplification = 30, 
                                fps = 4)


if vc.isOpened():
    is_capturing, _ = vc.read()
else:
    is_capturing = False
    
frames = []
changed_frames = []

while True:
    
    is_capturing, frame = vc.read()
        
    shape, img = face_detector.detect_faces(frame)
    
    if shape != 0:
        #leaves only face area
        #frame = cv2.rectangle(frame, shape[0], shape[1],
        #                     (255,0,0), thickness = 0) 
        
        # change colorspace
        #y_frame = EVM.BGR_YCrCb(frame)[:,:,0] 
        y_frame = frame
        
        EVM.frames.append(frame)
        EVM.apply_gaussian_pyramid(y_frame)
        #EVM.apply_laplacian_pyramid(y_frame) uncomment for motion
        
        if len(EVM.frames) > 20:
            amplified = EVM.magnify_color(y_frame, 0.4, 3)
            #amplified = EVM.magnify_motion(y_frame, 0.4, 3)
            
            changed_frames.append(amplified)
            cv2.imshow('img', cv2.convertScaleAbs(amplified))
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()

changed_frames = np.asarray(changed_frames, dtype = np.float64)

x = np.arange(0, len(changed_frames))

#plt.plot(changed_frames[2,100:300,200:400,0])

try:
    EVM.save_video(changed_frames)
    print('Success')
except Exception as e:
    print(e)
