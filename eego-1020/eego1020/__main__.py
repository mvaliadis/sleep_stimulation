# -*- coding: utf-8 -*-
"""
CLI
---

"""
from eego1020.streamer import Outlet, get_amplifiers, scan
import argparse
# %%
if __name__ == '__main__':
    desc = """
    Stream EEGO EEG amplifiers with LSL"""
    
    epilog = """
    If you can not find any amplifiers there are two possible reasons:
    
    1. Bad USB connection
        Unplug and reconnect the amplifier
    2. Driver Issues
        Reinstall the drivers appropriate for your OS from the small white
        USB-stick that came with the amplifiers.
        
        The drivers for WIN10 should be in the subfolder
        '\\eego-SDK-<serialnumber>\\windows\\driver\\win8\\x64'
        Right-click on the file `cyusb3.inf`and left-click on `Install`
        
        Rarely, the driver can become corrupted. In that case, uninstall it from
        the control panel / devices etc.    
    """
    parser = argparse.ArgumentParser(description=desc, epilog=epilog)
    parser.add_argument('--scan', action='store_true',
                        help='report the available devices')
    
    parser.add_argument('--nominal_srate', type=int, default=512,
                        help='''The preferred sampling rate.''')
    
    args = parser.parse_args()
    if args.scan:
        scan() 
    else:
        try:
            amp = get_amplifiers()[0]
            outlet = Outlet(amp, nominal_srate=args.nominal_srate)
            running = False
            while not running:
                try:
                    outlet.start()
                    running = True
                except RuntimeError:
                    pass
        except IndexError:
            pass
