import serial
from serial.tools.list_ports import comports

from torchvision.utils import save_image

import numpy as np
from PIL import Image
from io import BytesIO
from utils import detect_and_crop_faces

import time
import argparse
import os

baudRate = 460800
devices = []

def print_until_keyword(keyword, dev):
    while True: 
        msg = dev.readline().decode()
        if msg[:-2] == keyword:
            break
        else:
            print(f'({dev.port}):',msg, end='')

def read_port(port):
    while True:
        try:
            dev =  serial.Serial(None, baudRate)
            dev.port = port
            dev.rts = False
            dev.dtr = False
            dev.open()
            time.sleep(1)
            dev.rts = False
            dev.dtr = False
            time.sleep(1)
            dev.reset_input_buffer()
            print_until_keyword('d', dev)
            dev.write(b'd')
            pos = dev.readline().decode()[:-2]
            print(pos, port)
            return (pos, dev)
        except Exception as error:
            print("An exception occurred: ", error) # An exception occurred: division by zero

def read_label(msg):
    while True:
        try:
            label = input(msg)
            return label
        except:
            print(f"ERROR: Not a string ({label})")

def count_files_in_folder(folder_path):
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

def jpeg_buffer_to_rgb888(jpeg_buffer):
    # Decode JPEG buffer
    img = Image.open(BytesIO(jpeg_buffer))
    img_array = np.array(img)
    img_rgb888 = Image.fromarray(img_array)

    return img_rgb888

def getDatas(label, tDevice):
    _, dev = tDevice
    dev.write(b's')
    len = int(dev.readline().decode()[:-2])
    # print(len)
    buf = np.frombuffer(dev.read(len), dtype=np.uint8)

    # buf = np.reshape(buf, (1200,1600,3))
    if not os.path.exists(f'{args.face_dataset}/{label}'):
        os.makedirs(f'{args.face_dataset}/{label}')

    # print(buf)
    image = jpeg_buffer_to_rgb888(buf)
    num = count_files_in_folder(f'{args.face_dataset}/{label}') + 1
    save_image(detect_and_crop_faces(image), f'{args.face_dataset}/{label}/{num}.png')
    

#________________________ START ___________________________
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get face's image from devices")
    parser.add_argument(
        "--face_dataset",
        type=str,
        default="./face_dataset",
        help=f"Directory store images! (deafault 'dataset')",
    )
    args = parser.parse_args()

    dev = read_port(comports()[0].device)


    while True:
        label = read_label('Label: ')
        getDatas(label, dev)