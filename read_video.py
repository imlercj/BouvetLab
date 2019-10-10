#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:57:10 2019

@author: christoph.imler
"""

 import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile as wav
import time
from matplotlib import pyplot as plt

file_name_video = 'temp_video.avi'
file_name_audio = 'temp_audio.wav'


vidcap = cv2.VideoCapture(file_name_video)
success,image = vidcap.read()
count = 0

#while success:  
#  success,image = vidcap.read()
#  print('Read a new frame: ', success)
#  if success:
#      plt.imshow(image)
#      break
#  count += 1

def get_frame(seconds, file_name_video):
    vidcap = cv2.VideoCapture(file_name_video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(fps)
    frame_nr = int(fps*seconds)
    count = 0
    sec = 0 
    #plt.ion()
    while True:
        success,image = vidcap.read()
        sec += vidcap.get(cv2.CAP_PROP_POS_MSEC)/1000
        print('seconds', sec)
        #plt.imshow(image)
        #plt.pause(0.005)
        
        count += 1 
        if sec >=seconds:
            break
    plt.imshow(image)


def wav_plotter(full_path):
    rate, wav_sample = wav.read(full_path)
    print('sampling rate: ',rate,'Hz')
    print('number of channels: ',wav_sample.shape)
    print('duration: ',wav_sample.shape[0]/rate,' second')
    print('number of samples: ',len(wav_sample))
    plt.figure(figsize=(12, 4))
    plt.plot(wav_sample) 
    plt.show()
    return rate, wav_sample

if __name__ == '__main__':
    rate, wave = wav_plotter(file_name_audio)
    ss = pd.Series(wave, index=np.arange(0,len(wave)/rate,1/rate))
    plt.figure(figsize=(12, 4))
    plt.plot(ss.apply(lambda x:x**2).rolling(1000).sum()) 
    plt.show()
    max_seconds = ss.apply(lambda x:x**2).rolling(1000,center=True).sum().idxmax()
    print(max_seconds)
    get_frame(max_seconds, file_name_video)
    