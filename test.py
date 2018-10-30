import numpy as np
import cv2
from skimage.measure import compare_ssim

def check_YCbCr_YUV():
    vc = cv2.VideoCapture(0)
    
    if vc.isOpened():
        is_capturing, _ = vc.read()
    else:
        is_capturing = False
    
    while True:
        is_capturing, frame = vc.read()
            
        # shape, img = face_detector.detect_faces(frame)
        
        #if shape != 0:
         #   frame = cv2.rectangle(frame, shape[0], shape[1],
          #                        (255,0,0), thickness = 0)
           
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) 
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV) 
        
        cv2.imshow('YCrCb[0]',frame1[:,:,0])
        cv2.imshow('YCrCb[1]',frame1[:,:,1])
        cv2.imshow('YCrCb[2]',frame1[:,:,2])
        
        cv2.imshow('YUV[0]',frame2[:,:,0])
        cv2.imshow('YUV[1]',frame2[:,:,1])
        cv2.imshow('YUV[2]',frame2[:,:,2])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    vc.release()
    cv2.destroyAllWindows()

def thresholded():
    cap = cv2.VideoCapture(0)
    ret, frame_old = cap.read()
    
    while True:
        
        frame_old_gray = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
        
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        (score, diff) = compare_ssim(frame_old_gray, frame_gray, full=True)
        
        diff[diff > 0.2] = 1
        diff = (diff * 255).astype("uint8")
        
        #ret, thresh = cv2.threshold(diff, 0,50,cv2.THRESH_BINARY)
        thresh = cv2.threshold(diff, 0, 255,
                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        cv2.imshow('frame',thresh)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_old = np.copy(frame)
        
    cap.release()
    cv2.destroyAllWindows()

def test_colors():
    cap = cv2.VideoCapture(0)
    ret, frame_old = cap.read()
    
    while True:
    
        frame_old_r =frame_old[:,:,2]
        frame_old_g =frame_old[:,:,1]
        frame_old_b =frame_old[:,:,0]
        
        ret, frame = cap.read()
        
        frame_r =frame[:,:,2]
        frame_g =frame[:,:,1]
        frame_b =frame[:,:,0]
    
        (score, diff1) = compare_ssim(frame_old_r, frame_r, full=True)
        (score, diff2) = compare_ssim(frame_old_g, frame_g, full=True)
        (score, diff3) = compare_ssim(frame_old_b, frame_b, full=True)
        
        #diff[diff > 0.2] = 1
        diff1 = (diff1 * 255).astype("uint8")
        diff2 = (diff2 * 255).astype("uint8")
        diff3 = (diff3 * 255).astype("uint8")
        
        ret, thresh = cv2.threshold(diff, 0,50,cv2.THRESH_BINARY)
        thresh1 = cv2.threshold(diff1, 0, 255,
                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh2 = cv2.threshold(diff2, 0, 255,
                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh3 = cv2.threshold(diff3, 0, 255,
                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        cv2.imshow('red',thresh1)
        cv2.imshow('green',thresh2)
        cv2.imshow('blue',thresh3)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_old = np.copy(frame)
        
    cap.release()
    cv2.destroyAllWindows()
    
cap = cv2.VideoCapture(0)
ret, frame_old = cap.read()

while True:

    frame_old_r =frame_old[:,:,2]
    
    ret, frame = cap.read()
    
    frame_r = frame[:,:,2]
    
    (score, diff1) = compare_ssim(frame_old_r, frame_r, full=True)
    
    diff1[diff1 < 0.8] = 1
    diff1 = (diff1 * 255).astype("uint8")
    
    thresh1 = cv2.threshold(diff1, 0, 255,
                         cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cv2.imshow('red',thresh1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_old = np.copy(frame)
    
cap.release()
cv2.destroyAllWindows()
