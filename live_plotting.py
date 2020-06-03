#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:43:36 2020

@author: mariuskeute
"""

"""
pulls EEG data from an LSL stream and plots it in real time
"""


from pylab import figure, show
import numpy as np
import liesl
from reiz import clock
import matplotlib.pyplot as plt

def multichannel_plot(datarray, chan_labels = None):    
    
    fig = figure()
    
    yprops = dict(rotation=0,
                  horizontalalignment='right',
                  verticalalignment='center')
    
    axprops = dict(yticks=[])
    nchans = np.shape(datarray)[1]
    ht = .9/nchans
    ypos = [0.7, 0.5, 0.3, 0.1]
    ypos = np.linspace(1-ht, .1, nchans)
    if chan_labels == None:
        chan_labels = [f'chan {x}' for x in range(nchans)]
        
    l=[]    
    for ch in range(nchans):
        ax = fig.add_axes([0.1, ypos[ch], 0.8, ht], **axprops)
        l.append(ax.plot(datarray[:,ch])[0])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(chan_labels[ch], **yprops)

        axprops['sharex'] = ax
        axprops['sharey'] = ax
    show()

    return fig, l


sinfo = liesl.get_streaminfos_matching(type = 'EEG')
bfr = liesl.RingBuffer(sinfo[0], duration_in_ms = 10000)
bfr.await_running()



datarray = np.random.rand(*bfr.max_shape)
fig, lines = multichannel_plot(datarray)
fig.canvas.draw()
fig.canvas.flush_events()
clock.sleep(10)

while plt.fignum_exists(1):
    datarray = bfr.get_data()
    
    for l in range(len(lines)):
        lines[l].set_ydata(datarray[:,l])
        plt.gca().set_ylim(min(datarray[:,l]),max(datarray[:,l]))
        clock.sleep(.1)
    fig.canvas.draw()
    fig.canvas.flush_events()
    # clock.sleep(.1) 