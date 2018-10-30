# -*- coding: utf-8 -*-
import cv2 
from skimage.transform import resize
import numpy as np
from scipy import fftpack, signal

class detectFace(object):
        
    def __init__(self, num_people = 1, margin = 10, casscade = 'haarcascade.xml'):
        self.margin = margin
        self.num_people = num_people
        self.casscade = cv2.CascadeClassifier(casscade)
    
    def detect_faces(self, frame):
        
        faces = self.casscade.detectMultiScale(frame,
                                         scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(100, 100))
        
        if len(faces) != 0:
            face = faces[0]
            (x,y,w,h) = face
            left = x - self.margin // 2
            right = x + w + self.margin // 2
            bottom = y - self.margin // 2
            top = y + h + self.margin // 2
            
            shape = [(left - 1, bottom - 1), (right + 1, top + 1)]
        
            img = resize(frame[bottom:top, left:right, :],
                         (160,160), mode ='reflect')

        else:
            shape = 0
            img = None
            
        return shape, img

class Euler_Video_Magnification(object):
    
    def __init__(self,
                 level, 
                 amplification, 
                 fps,
                 colorspace = 'YcbCr',
                 backward_frames = 15):
        
        self.frames = []
        self.pyramids = []
        self.laplacian_pyramids = [[] for i in range(level)]
        self.colorspace = colorspace
        self.level = level
        self.amplification = amplification
        self.fps = fps
        self.backward_frames = backward_frames
        
    def BGR_YCrCb(self,
                  frame):
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        self.frames.append(frame)
        
        return frame
        
    def YCrCb_BGR(self,frame):
        
        frame = cv2.cvtColor(frame, cv2.COLOR_YCrCb2BGR)
        
        return frame    

    def gaussian_pyramid(self, 
                         frame):
        
        subsample = np.copy(frame)
        pyramid_list = [subsample]
        
        for i in range(self.level):
            subsample = cv2.pyrDown(subsample)
            pyramid_list.append(subsample)

        return pyramid_list

    def build_gaussian_pyramid(self, 
                               tensor):
        
        frame = tensor
        pyr = self.gaussian_pyramid(frame)
        gaussian_frame=pyr[-1]
        tensor_data = gaussian_frame
        
        return tensor_data

    def laplacian_pyramid(self, 
                          frame):
        
        gaussian_pyramids = self.gaussian_pyramid(frame)
        laplacian_pyramids = []
        
        for i in range(self.level, 0, -1):
            upper = cv2.pyrUp(gaussian_pyramids[i])
            sample = cv2.subtract(gaussian_pyramids[i-1], upper)
            laplacian_pyramids.append(sample)
            
        return laplacian_pyramids
        
    #bandpass filter    
    def bandpass_filter(self, 
                        tensor, 
                        low, 
                        high, 
                        axis = 0):
        
        frames_arr = np.asarray(tensor, dtype = np.float64)
        fft = fftpack.fft(frames_arr, axis = axis)
        frequencies = fftpack.fftfreq(frames_arr.shape[0], d = 1.0 / self.fps)
        bound_low = (np.abs(frequencies - low)).argmin()
        bound_high = (np.abs(frequencies - high)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0
        
        iff = fftpack.ifft(fft, axis = axis)
        
        return np.abs(iff)
    
    def amplify_frame(self, 
                      frame):
        return frame * self.amplification
    
    def reconstruct_video(self, 
                          amp_video, 
                          original_video):
        final_video = np.zeros(original_video.shape)
        for i in range(0,amp_video.shape[0]):
            img = amp_video[i]
            for x in range(self.level):
                img=cv2.pyrUp(img)
            img=img+original_video[i]
            final_video[i]=img
        return final_video
    
    def reconstruct_frame(self, 
                          amp_frame, 
                          original_frame):
        final_video = np.zeros(original_frame.shape)
        img = amp_frame
        for x in range(self.level):
            img = cv2.pyrUp(img)
        img = img+original_frame
        final_video = img
        return final_video
    
    def magnify_color(self, 
                      frame, 
                      low, 
                      high):
        
        filtered = self.bandpass_filter(self.pyramids[-self.backward_frames:], low, high)
        amplified_frames = self.amplify_frame(filtered)
        final = self.reconstruct_frame(amplified_frames[-1], frame)
        return final
        
    def apply_gaussian_pyramid(self, 
                               frame):
        
        pyramid = self.build_gaussian_pyramid(frame)
        self.pyramids.append(pyramid)
    
    def apply_laplacian_pyramid(self,
                                frame):
        
        lp_pyramid = self.laplacian_pyramid(frame)
            
        for i in range(self.level):
            self.laplacian_pyramids[i].append(lp_pyramid[i])
        
    def convert_to_np(self):
        
        for i in range(self.level):
            self.laplacian_pyramids[i] = np.array(self.laplacian_pyramids[i], dtype = np.float64)
        
    def butter_bandpass_filter(self,
                               data, 
                               lowcut, 
                               highcut, 
                               fs,
                               order=5):
        
        omega = 0.5 * fs
        low = lowcut / omega
        high = highcut / omega
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.lfilter(b, a, data, axis=0)
        return y
        
    def reconstract_from_tensorlist(self,
                                    frame_tensor):
        
        final = np.zeros(frame_tensor[0][-1].shape)
        up = frame_tensor[0][-1]
        
        for i in range(self.level-1):
            up = cv2.pyrUp(up) + frame_tensor[i + 1][-1]
        final = up
        
        return final
        
    def magnify_motion(self,
                       frame,
                       low,
                       high):
        
        #self.convert_to_np()
        filter_tensor_list = []
        
        for i in range(self.level):
            np_laplacian = np.array(self.laplacian_pyramids[i][-3:], dtype = np.float64)
            tensor = self.butter_bandpass_filter(np_laplacian, low, high, self.fps)
            tensor *= self.amplification
            filter_tensor_list.append(tensor)
            
        recon = self.reconstract_from_tensorlist(filter_tensor_list)
        
        final = frame + recon
        
        return final

    def save_video(self, 
                   video_tensor):
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        [height,width]=video_tensor[0].shape[0:2]
        writer = cv2.VideoWriter("out_test.avi", fourcc, 30, (width, height), 1)
        for i in range(0,video_tensor.shape[0]):
            writer.write(cv2.convertScaleAbs(video_tensor[i]))
        writer.release()
        