import torch
import torchvision.transforms as transforms

import serial
from serial.tools.list_ports import comports

import numpy as np
from PIL import Image
from io import BytesIO

from model_PTH import MobileFaceNet
from utils import detect_and_crop_faces, cal_similarity

import argparse
import pickle
import paho.mqtt.client as mqtt
import json
import socket
import time
import base64

from typing import OrderedDict
from client import SAVED_CLIENT
from server import THRESHOLD
from get_data_client import BAUDRATE
from know_faces_embedding_client import SAVED_FILE

#________________________ VARIABLES ___________________________

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

def on_request(client: mqtt.Client, userdata, message):
    global esp32
    pi_id = message.payload.decode("utf-8")
    # image = Image.open('./face_dataset/20176752/7.png')
    id = None
    max_sim = THRESHOLD
    buffered = BytesIO()

    # Get face from esp32's image
    print_until_keyword('wait', esp32)
    esp32.write(b's')
    line = esp32.readline().decode()[:-2]
    if line.isdigit():
        len = int(line)
        buf = np.frombuffer(esp32.read(len), dtype=np.uint8)
        image = jpeg_buffer_to_rgb888(buf)
        image = detect_and_crop_faces(image)

        if image is not None:
            embedding = model(transform(image).unsqueeze(0)).cpu().detach().numpy().flatten()

            for person_id, known_embedding in faces_embedding.items():
                sim = cal_similarity(embedding, known_embedding)

                if sim > max_sim:
                    max_sim = sim
                    id = person_id

            pil_img = transforms.ToPILImage()(image)
            pil_img.save(buffered, format="JPEG")
    else:
        print(line)
    client.publish('raspberry_pi_response/face_recognize', payload=json.dumps({'pi_id': pi_id, 'data': {'score': float(max_sim), 'id': id, 'image': base64.b64encode(buffered.getvalue()).decode('utf-8') }}))

#________________________ START ___________________________
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="MQTT face recognize client!")
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

    print(socket.gethostname())
    port = next((port.device for port in comports() if 'ttyUSB' in port.device), None)
    if port is None:
        exit('No device connect!')
    _, esp32 = read_port(port)

    client = mqtt.Client(userdata={"hostname": socket.gethostname()})
    client.on_message = on_request
    client.connect("localhost", 1883)

    # Subscribe to the request topic
    client.subscribe('raspberry_pi_request/face_recognize')

    # Loop to handle incoming requests
    client.loop_forever()