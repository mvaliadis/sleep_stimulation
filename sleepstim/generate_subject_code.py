#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:45:00 2020

@author: mvaliadis
"""

from random import SystemRandom
sr = SystemRandom() # create an instance of the SystemRandom class
    

def generate_subject_code(length, 
                      valid_chars=None):
    """ generate_subject_code(length, check_char) -> subject code
        length: the length of the created subject code
        check_char: a Boolean function used to check the validity of a char
    """
    if valid_chars==None:
        valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        valid_chars += "0123456789"
    
    code = ""
    counter = 0
    while counter < length:
        rnum = sr.randint(0, 128)
        char = chr(rnum)
        if char in valid_chars:
            code += chr(rnum)
            counter += 1
    return code
