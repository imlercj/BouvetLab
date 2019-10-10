#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:26:23 2019

@author: christoph.imler
"""
import cv2
import pyaudio
import threading
import time
import subprocess
import os
import wave
import numpy as np
import pandas as pd
from scipy.io import wavfile as wav
from matplotlib import pyplot as plt

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

class VideoRecorder():  

    def change_res(self, cap, width, height):
        cap.set(3, width)
        cap.set(4, height)

    # grab resolution dimensions and set video capture to it.
    def get_dims(self, cap, res='1080p'):
        width, height = STD_DIMENSIONS["480p"]
        if res in STD_DIMENSIONS:
            width,height = STD_DIMENSIONS[res]
        ## change the current caputre device
        ## to the resulting resolution
        self.change_res(cap, width, height)
        return width, height
    
    def get_video_type(self, filename):
        filename, ext = os.path.splitext(filename)
        if ext in VIDEO_TYPE:
          return  VIDEO_TYPE[ext]
        return VIDEO_TYPE['mp4']
            
    
    # Video class based on openCV 
    def __init__(self):

        self.open = True
        self.device_index = 0
        self.frames_per_second = 48.0
        self.res = '720p'
        self.video_filename = "temp_video.avi"
        self.video_cap = cv2.VideoCapture(0)
        self.video_out = cv2.VideoWriter(self.video_filename, self.get_video_type(self.video_filename), self.frames_per_second, self.get_dims(self.video_cap, self.res))
        self.frame_counts = 1
        self.start_time = time.time()


    # Video starts being recorded 
    def record(self):

#       counter = 1
        timer_start = time.time()
        timer_current = 0


        while(self.open==True):
            ret, video_frame = self.video_cap.read()
            self.video_out.write(video_frame)
            #cv2.imshow('frame',video_frame)
            if (ret==True):

                    
#                   print str(counter) + " " + str(self.frame_counts) + " frames written " + str(timer_current)
                    self.frame_counts += 1
#                   counter += 1
#                   timer_current = time.time() - timer_start
                    time.sleep(1/self.frames_per_second)
#                   gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
#                   cv2.imshow('video_frame', gray)
#                   cv2.waitKey(1)
            else:
                print('break')
                #break

                # 0.16 delay -> 6 fps
                # 


    # Finishes the video recording therefore the thread too
    def stop(self):

        if self.open==True:

            self.open=False
            self.video_out.release()
            self.video_cap.release()

            cv2.destroyAllWindows()

        else: 
            pass


    # Launches the video recording function using a thread          
    def start(self):
        video_thread = threading.Thread(target=self.record)
        video_thread.start()





class AudioRecorder():


    # Audio class based on pyAudio and Wave
    def __init__(self):

        self.open = True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.audio_filename = "temp_audio.wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []


    # Audio starts being recorded
    def record(self):

        self.stream.start_stream()
        while(self.open == True):
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            if self.open==False:
                break


    # Finishes the audio recording therefore the thread too    
    def stop(self):

        if self.open==True:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

        pass

    # Launches the audio recording function using a thread
    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()





def start_AVrecording(filename='test'):

    global video_thread
    global audio_thread

    video_thread = VideoRecorder()
    audio_thread = AudioRecorder()

    audio_thread.start()
    video_thread.start()

    return filename




def start_video_recording(filename='test'):

    global video_thread

    video_thread = VideoRecorder()
    video_thread.start()

    return filename


def start_audio_recording(filename='test'):

    global audio_thread

    audio_thread = AudioRecorder()
    audio_thread.start()

    return filename




def stop_AVrecording(filename='test', initial_threads=1):
    print('initial_threads:', initial_threads, '\nthreads_active:', threading.active_count())
    audio_thread.stop() 
    frame_counts = video_thread.frame_counts
    elapsed_time = time.time() - video_thread.start_time
    recorded_fps = frame_counts / elapsed_time
    print ("total frames " + str(frame_counts))
    print ("elapsed time " + str(elapsed_time))
    print ("recorded fps " + str(recorded_fps))
    video_thread.stop() 

    # Makes sure the threads have finished
    #while threading.active_count() > initial_threads:
    #    print('threads_active:', threading.active_count())
        
   #     time.sleep(1)





# Required and wanted processing of final files
def file_manager(filename='test'):

    local_path = os.getcwd()

    if os.path.exists(str(local_path) + "/temp_audio.wav"):
        os.remove(str(local_path) + "/temp_audio.wav")

    if os.path.exists(str(local_path) + "/temp_video.avi"):
        os.remove(str(local_path) + "/temp_video.avi")

    if os.path.exists(str(local_path) + "/temp_video2.avi"):
        os.remove(str(local_path) + "/temp_video2.avi")

    if os.path.exists(str(local_path) + "/" + filename + ".avi"):
        os.remove(str(local_path) + "/" + filename + ".avi")


## Analyse
        
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
    print('start main')
    file_name_video = 'temp_video.avi'
    file_name_audio = 'temp_audio.wav'
    
    initial_threads = threading.active_count()
    start_AVrecording()
    time.sleep(5)
    stop_AVrecording(initial_threads=initial_threads)
    
    ## Analyse
    print('start analysis')
    rate, wave_ = wav_plotter(file_name_audio)
    ss = pd.Series(wave_, index=np.arange(0,len(wave_)/rate,1/rate))
    plt.figure(figsize=(12, 4))
    plt.plot(ss.apply(lambda x:x**2).rolling(1000).sum()) 
    plt.show()
    max_seconds = ss.apply(lambda x:x**2).rolling(1000,center=True).sum().idxmax()
    print('max second', max_seconds)
    get_frame(max_seconds, file_name_video)
    
    
    file_manager()
    
    