#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:46:36 2021

@author: administrator
"""

from cryptography.fernet import Fernet
from cryptography.fernet import Fernet, InvalidToken
import numpy as np

input_file = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes.csv'
output_file = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes_encrypted.csv'

def encrypt_file(input_file, output_file):
    key = Fernet.generate_key() 
    with open(input_file, 'rb') as f:
        data = f.read()  # Read the bytes of the input file
        
    fernet = Fernet(key)
    encrypted = fernet.encrypt(data)

    with open(output_file, 'wb') as f:
        f.write(encrypted)  # Write the encrypted bytes to the output file
    return key

def decrypt_file(encrypt_key, input_file, output_file, decrypt_save=False):
    # encrypt_key must be the same as used in encrypting...    
    with open(input_file, 'rb') as f:
        data = f.read()  # Read the bytes of the encrypted file
    
    fernet = Fernet(encrypt_key)
    try:
        decrypted = fernet.decrypt(data).splitlines()[1::]
        decrypted_data = np.asarray([decrypted[i].decode().split(',') for i in range(len(decrypted))])
        
        if decrypt_save:       
            with open(output_file, 'wb') as f:
                f.write(decrypted)  # Write the decrypted bytes to the output file
    
        # Note: You can delete input_file here if you want
    except InvalidToken as e:
        print("Invalid Key - Unsuccessfully decrypted")
        
    return decrypted_data
        
if __name__ == '__main__':
    encrypt_key = encrypt_file(input_file, output_file)
    decrypted = decrypt_file(encrypt_key, input_file=output_file, output_file=input_file)
