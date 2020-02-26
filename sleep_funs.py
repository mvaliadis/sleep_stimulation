# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:58:01 2020

@author: neuro
"""



import pyaudio
import numpy as np
import pandas as pd
from reiz import clock
import reiz
import threading

class PinkNoise():
    def generate_noise(self,duration_in_s = 0.05, fs = 48000, ncols=16):
        """Generates pink noise using the Voss-McCartney algorithm.
        
        nrows: number of values to generate
        rcols: number of random sources to add
        
        returns: NumPy array
        """
        nrows = int(fs * duration_in_s)
        array = np.empty((nrows, ncols))
        array.fill(np.nan)
        array[0, :] = np.random.random(ncols)
        array[:, 0] = np.random.random(nrows)
        
        # the total number of changes is nrows
        n = nrows
        cols = np.random.geometric(0.5, n)
        cols[cols >= ncols] = 0
        rows = np.random.randint(nrows, size=n)
        array[rows, cols] = np.random.random(n)
        
        flank_samples = int(0.005*fs)
        rising = np.linspace(0,1,flank_samples)
        falling = np.linspace(1,0,flank_samples)
        plateau = np.ones(nrows-2*flank_samples)
        window = np.concatenate((rising,plateau,falling))
        df = pd.DataFrame(array)
        df.fillna(method='ffill', axis=0, inplace=True)
        total = df.sum(axis=1)
        noise = total.values
        noise /= max(noise)
        return noise*window
    
    
    def open_stream(self):  
        fs = 48000
        self.stream = self.p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=fs,
                        output=True)
        
    def play(self):
        volume = 1     # range [0.0, 1.0]
        if self.reizmarker:
            reiz.marker.push('pinknoise')
        self.stream.write(volume*self.samples)
        self.stream.stop_stream()
        self.stream.start_stream()

#        stream.close()
#        return t
#        self.p.terminate()
        
    def __init__(self):
        threading.Thread.__init__(self)
        self.samples = self.generate_noise()
        self.p = pyaudio.PyAudio()
        self.open_stream()
        self.reizmarker = True
        if not reiz.marker.available():
            print('Marker Server not available!')
            self.reizmarker = False
    
