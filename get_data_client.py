import serial
from serial.tools.list_ports import comports

from torchvision.utils import save_image

import numpy as np
from PIL import Image
from io import BytesIO
from utils import detect_and_crop_faces
import base64

import time
import argparse
import os
import datetime
# import matplotlib.pyplot as plt

BAUDRATE = 230400
devices = []

def print_until_keyword(keyword, dev):
    while True: 
        msg = dev.readline().decode()
        if msg[:-2] == keyword:
            break
        else:
            print(f'({dev.port}):',msg, end='')

def read_port(port):
    try:
        dev =  serial.Serial(None, BAUDRATE, timeout = 10.0)
        dev.port = port
        dev.rts = False
        dev.dtr = False
        dev.open()
        time.sleep(1)
        dev.rts = False
        dev.dtr = False
        time.sleep(1)
        dev.reset_input_buffer()
        # print_until_keyword('d', dev)
        # dev.write(b'd')
        # pos = dev.readline().decode()[:-2]
        # print(pos, port)
        return dev
    except Exception as error:
        print("An exception occurred: ", error)
        return None

def read_string(msg):
    while True:
        try:
            label = input(msg)
            if label is not None or len(label) != 0:
                return label
        except:
            print(f"ERROR: Not a string ({label})")
            continue
            

def jpeg_buffer_to_rgb888(jpeg_buffer):
    # Decode JPEG buffer
    img = Image.open(BytesIO(jpeg_buffer))
    img_array = np.array(img)
    img_rgb888 = Image.fromarray(img_array)

    return img_rgb888

def getDatas(label, dev):
    if not os.path.exists(f'{args.face_dataset}/{label}'):
        os.makedirs(f'{args.face_dataset}/{label}')
    try:
        print_until_keyword('wait', dev)
        dev.write(b's')
        line = dev.readline().decode()[:-2]
        if line.isdigit():
            len = int(line)
            buf = np.frombuffer(dev.read(len), dtype=np.uint8)
            image = jpeg_buffer_to_rgb888(buf)
            # plt.imshow(np.asarray(image))
            # plt.show()
            image = detect_and_crop_faces(image)
            if image is not None: save_image(image, f'{args.face_dataset}/{label}/{datetime.datetime.now()}.png')
    except Exception as error:
        print("An exception occurred: ", error)
        return False
    
    return True
    

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

    port = next((port.device for port in comports() if 'ttyUSB' in port.device), None)
    if port is None:
        exit('No device connect!')
    dev = read_port(port)
    if dev is None: exit('Can\'t connect to serial port!')

    while True:
        label = read_string('Label: ')
        if label == 'exit':
            break
        read_time = time.time()
        getDatas(label, dev)
        print(f'Time: {time.time() - read_time}')
        print('done')