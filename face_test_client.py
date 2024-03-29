import torch
import torchvision.transforms as transforms

import serial
from serial.tools.list_ports import comports

import numpy as np
from PIL import Image
from io import BytesIO

from model_PTH import MobileFaceNet
from utils import detect_and_crop_faces, cal_similarity

import time
import argparse
import pickle

from typing import OrderedDict
from client import SAVED_CLIENT
from know_faces_embedding_client import SAVED_FILE

#________________________ VARIABLES ___________________________
BAUDRATE = 460800

#________________________ FUNCTION ___________________________
def print_until_keyword(keyword, dev):
    while True: 
        msg = dev.readline().decode()
        if msg[:-2] == keyword:
            break
        else:
            print(f'({dev.port}):',msg, end='')

def read_port(port):
    try:
        dev =  serial.Serial(None, BAUDRATE)
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
        print("An exception occurred: ", error)
        return None

def jpeg_buffer_to_rgb888(jpeg_buffer):
    img = Image.open(BytesIO(jpeg_buffer))
    img_array = np.array(img)
    img_rgb888 = Image.fromarray(img_array)

    return img_rgb888

#________________________ START ___________________________
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify face")
    parser.add_argument(
        "--saved_file",
        type=str,
        default=SAVED_FILE,
        help="Location of saved embedding file! (default: know_faces_embedding.pth)",
    )
    args = parser.parse_args()

    device = torch.device('cpu')
    model = MobileFaceNet().to(device)

    state_dict = torch.load(SAVED_CLIENT + '/best_model.pth', map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # Loading the saved embeddings
    # faces_embedding = torch.load(SAVED_CLIENT + args.saved_file)
    with open(SAVED_CLIENT + args.saved_file, 'rb') as handle:
        faces_embedding = pickle.load(handle)

    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]),
    ])

    _, dev = read_port(comports()[0].device)
    while True:
        cmd = input('Face_verify_input:')
        if cmd == 'exit':
            break
        # Get face from esp32's image
        dev.write(b's')
        len = int(dev.readline().decode()[:-2])
        buf = np.frombuffer(dev.read(len), dtype=np.uint8)
        image = jpeg_buffer_to_rgb888(buf)
        image = detect_and_crop_faces(image)

        # image = Image.open('./face_dataset/m/20176752/7.png')

        embedding = model(transform(image).unsqueeze(0)).cpu().detach().numpy().flatten()

        id = None
        max_sim = 0.9

        # print(faces_embedding)

        for person_id, known_embedding in faces_embedding.items():
            sim = cal_similarity(embedding, known_embedding)
            # print(sim)

            if sim > max_sim:
                max_sim = sim
                id = person_id

        print(f'Score: {max_sim}')
        print(f'ID: {id}')