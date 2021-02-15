# -*- coding: utf-8 -*-
"""
Streamer
--------

Backend to the EEGO SDK

"""
from pylsl import StreamInfo, StreamOutlet, local_clock
import time
import threading
import eego.eego_sdk as sdk
import pathlib
# from eego.waveguard.cap import get_channel_names
# %%

def get_channel_names():
    names = ['Fpz', 'Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'O1', 'O2', 'M1', 'M2', 'bipEOG', 'bipEMG', 'bipECG']
    with open(pathlib.Path(__file__).parent / 'waveguard64.txt') as f:
        elec = f.readlines()
    chnames = [str_.split(':')[0] for str_ in elec]
    
    idx = []
    for val2 in names:
        for ix,val in enumerate(chnames):
            if val2 == val:
                idx.append(ix)
    idx.extend(list(range(76,79)))
    return names, idx


def get_amplifiers():
    factory = sdk.factory()
    #v = factory.getVersion()
    #print('version: {}.{}.{}.{}'.format(v.major, v.minor, v.micro, v.build))
    amps =factory.getAmplifiers()
    if len(amps) == 0:
        print("No amplifiers found")
    for amp in amps:
        print('Found', amplifier_to_id(amp))
    return amps

def amplifier_to_id(amp):
  return '{}-{:06d}-{}'.format(amp.getType(), amp.getFirmwareVersion(), amp.getSerialNumber())

def properties(amp):
    rates = amp.getSamplingRatesAvailable()
    ref_ranges = amp.getReferenceRangesAvailable()
    bip_ranges = amp.getBipolarRangesAvailable()
    print('amplifier: {}'.format(amplifier_to_id(amp)))
    print('  rates....... {}'.format(rates))
    print('  ref ranges.. {}'.format(ref_ranges))
    print('  bip ranges.. {}'.format(bip_ranges))
 #   print('  channels.... {}'.format(amp.getChannelList()))
 
def scan():
    amps = get_amplifiers()
    for amp in amps:
        properties(amp)
# %%
def get_version():
    factory = sdk.factory()
    v = factory.getVersion()
    return 'version: {}.{}.{}.{}'.format(v.major, v.minor, v.micro, v.build)

# %%
def create_receiver(amp, nominal_srate=None, ref_range=None, bip_range=None):
    if bip_range is None:
        bip_range = amp.getBipolarRangesAvailable()[0]
    if ref_range is None:
        ref_range = amp.getReferenceRangesAvailable()[0]
    if nominal_srate is None:
        nominal_srate = amp.getSamplingRatesAvailable()[0]
        
    receiver = amp.OpenEegStream(nominal_srate, ref_range, bip_range)
    return receiver, nominal_srate
    

def create_outlet(amp, nominal_srate=None, ref_range=None, bip_range=None):
    
    receiver, nominal_srate = create_receiver(amp,  nominal_srate, ref_range, bip_range)
    source_id = amplifier_to_id(amp)
    channel_count = len(amp.getChannelList())    
    info = StreamInfo(name='eego', type='EEG', 
                      channel_count =channel_count, 
                      nominal_srate=nominal_srate, 
                      channel_format='float32',
                      source_id=source_id)

    # append some meta-data
    info.desc().append_child_value("manufacturer", "eego")
    channels = info.desc().append_child("channels")
    for c in get_channel_names(channel_count):
        channels.append_child("channel") \
            .append_child_value("label", c) \
            .append_child_value("unit", "microvolts") \
            .append_child_value("type", "EEG")
                
    outlet = StreamOutlet(info, chunk_size=50, max_buffered=360)
    
    while True:
        data = receiver.getData()
        sample_count= data.getSampleCount()
        chunk = []
        for sample_idx in range(sample_count):
            sample = []
            for chan_idx in range(channel_count):
                sample.append(data.getSample(chan_idx, sample_idx))
            chunk.append(sample)
        outlet.push_chunk(chunk)

class Outlet(threading.Thread):
    
    def __init__(self, amp, nominal_srate=None, ref_range=None, bip_range=None):
        threading.Thread.__init__(self)
        self.receiver, nominal_srate = create_receiver(amp, nominal_srate, 
                                                       ref_range, bip_range)
        source_id = amplifier_to_id(amp)
        chanlist = get_channel_names()
        self.chanix = chanlist[1]
        channel_count = len(self.chanix)
        self.info = StreamInfo(name='eego', type='EEG', 
                               channel_count =channel_count, 
                               nominal_srate=nominal_srate, 
                               channel_format='float32',
                               source_id=source_id)
    
        # append some meta-data
        self.info.desc().append_child_value("manufacturer", "eego")
        channels = self.info.desc().append_child("channels")
        for c in chanlist[0]:
            channels.append_child("channel") \
                .append_child_value("label", c) \
                .append_child_value("unit", "microvolts") \
                .append_child_value("type", "EEG")
                    
    def run(self):
        outlet = StreamOutlet(self.info)        
        counter = 0.
        chan_count = self.info.channel_count()  
        nominal_srate = self.info.nominal_srate()          
        print(self.info.as_xml())
        self.is_running = True        
        while self.is_running:
            data = self.receiver.getData()
            sample_count= data.getSampleCount()
            counter += sample_count                       
            chunk = []
            for sample_idx in range(sample_count):
                sample = []
                for chan_idx in self.chanix:
                    sample.append(data.getSample(chan_idx, sample_idx))
                chunk.append(sample)
            
            outlet.push_chunk(chunk)
            
            if counter>nominal_srate:
                counter = 0.
                print('Example sample:', ['{:3.2f}'.format(s) for s in sample], 'at', local_clock())
            
