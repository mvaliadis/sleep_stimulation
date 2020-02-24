# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:36:25 2020

@author: neuro
"""
from os import chdir
chdir('C:/Users/neuro/Documents/sleep_stimulation')
import liesl
from sleep_funs import PinkNoise
import matplotlib.pyplot as plt
#from reiz import clock
#%% set parameters
thresh = 0.9 #threshold to detect SO, should be set to -35 later
winshift_in_ms = 20

#%% set up EEG stream
sinfo = liesl.get_streaminfo_matching(type = 'EEG')
bfr = liesl.RingBuffer(sinfo, duration_in_ms = 2000, fs = 100)
bfr.start()



#TODO process EEG (filter? common avg. ref?)
# b,a = signal.butter(4, (.5, 1.5), btype = 'bandpass', fs = bfr.fs)
# t = signal.filtfilt(b,a,bfr.get_data(), axis = 0)

#TODO set up sleep stage classifier, gate SO detection
#TODO send marker
#TODO set up SO detection
t = bfr.get_data()[:,0] #pick one of the mock channels

#TODO send marker
#set up a pink noise burst generator
n = PinkNoise()
#play sound burst
tm = np.zeros(100)
for tr in range(100):
    tm[tr] = n.play()

#    
#import wave, struct, math, random
#sampleRate = 44100.0 # hertz
#duration = 1.0 # seconds
#frequency = 440.0 # hertz
#obj = wave.open('sound.wav','w')
#obj.setnchannels(1) # mono
#obj.setsampwidth(2)
#obj.setframerate(sampleRate)
#for i in range(40000):
#   value = random.randint(-32767, 32767)
#   data = struct.pack('<h', value)
#   obj.writeframesraw( data )
#obj.close()
##
#import pyglet
#explosion = pyglet.media.load('sound.wav', streaming=False)
#p = pyglet.media.Player()
#p.queue(explosion)
#
#p.queue([])
#p.play()
#p.pause()