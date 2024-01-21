import paho.mqtt.client as mqtt
import json
import time
import argparse
import tkinter as tk
from PIL import Image, ImageTk
import base64
from threading import Thread
from io import BytesIO

#____________________________________ VARIABLES ___________________________
HOSTNAME_LEFT = 'raspberrypi-left.local'
HOSTNAME_MID = 'raspberrypi-mid.local'
HOSTNAME_RIGHT = 'raspberrypi-right.local'
RES_TIMEOUT = 5
RECONNECT_TIME = 60

#________________________ FUNCTION ___________________________
def on_connect(client, userdata, flags, rc):
    pi_id = userdata["pi_id"]
    if rc == 0:
        print(f"Connected to {pi_id}.")
        client.subscribe('raspberry_pi_response/face_recognize')
    else:
        print(f"Failed to connect to {pi_id}.")

def on_message(client, userdata, message):
    global aggregated_results
    command = message.topic.split("/")[-1]
    if command == 'face_recognize':
        data = json.loads(message.payload.decode("utf-8"))
        aggregated_results[data['pi_id']] = data['data']

def reconnect_to_pi(pi_id):
    global clients
    print(f"Reconnecting to {pi_id}...")
    clients[pi_id].reconnect()

def update_images(sources: dict):
    while True:
        for image_source in sources.values():
            if image_source['image_io'] is not None:
                image = Image.open(image_source['image_io'])
            else:
                image = Image.new("RGB", (96, 112), "white")

            photo = ImageTk.PhotoImage(image)
            image_source['label'].config(image=photo)
            image_source['label'].image = photo
            time.sleep(0.1)

def get_face_recognize():
    global aggregated_results, image_sources
    while True:
        aggregated_results = {}
        for pi_id in pi_ids:
            if not clients[pi_id].is_connected():
                reconnect_to_pi(pi_id)
                clients[pi_id].loop_start()
            clients[pi_id].publish('raspberry_pi_request/face_recognize', payload=f"{pi_id}")

        start_time = time.time()
        while True:
            received_pis = set(aggregated_results.keys())

            if set(received_pis) == set(pi_ids):
                break

            if time.time() - start_time > RES_TIMEOUT:
                print("Timeout: Not all Raspberry Pi responded within the timeout period.")
                break
            time.sleep(0.1)


        if len(pi_ids) != len(aggregated_results):
            print('Some camera cannot verify face!')
            continue

        res = {
            'score': [],
            'id': [],
        }
        for pi_id, data in aggregated_results.items():
            res['score'].append(data['score'])
            res['id'].append(data['id'])
            image_sources[pi_id]['image_io'] = BytesIO(base64.b64decode(data['image']))
        

        if len(set(res['id'])) != 1:
            print('The face cannot be verify because the result of cameras is not match!')
            continue
        print(f'The face\'s id is {res["id"][0]} and the average score is {sum(res["score"]) / len(res["score"])}')
        print(f'Time: {time.time() - start_time}')
#________________________ START ___________________________
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="MQTT face recognize server!")
    parser.add_argument(
        "--pi_left",
        type=str,
        default=HOSTNAME_LEFT,
        help=f"The Raspberry Pi left's IP address or local hostname! (deafault {HOSTNAME_LEFT})",
    )
    parser.add_argument(
        "--pi_mid",
        type=str,
        default=HOSTNAME_MID,
        help=f"The Raspberry Pi mid's IP address or local hostname! (deafault {HOSTNAME_MID})",
    )
    parser.add_argument(
        "--pi_right",
        type=str,
        default=HOSTNAME_RIGHT,
        help=f"The Raspberry Pi right's IP address or local hostname! (deafault {HOSTNAME_RIGHT})",
    )
    args = parser.parse_args()


    pi_ids = [args.pi_left, args.pi_mid, args.pi_right]
    # pi_ids = ['localhost']
    image_sources = {pid_id: {'image_io': None, 'label': None} for pid_id in pi_ids}
    clients = {}
    aggregated_results = {}

    root = tk.Tk()
    root.title("Image Streaming Window")

    for i, pi_id in enumerate(pi_ids):
        label = tk.Label(root)
        label.grid(row=0, column=i, padx=5, pady=5)
        image_sources[pi_id]['label'] = label

    for pi_id in pi_ids:
        client = mqtt.Client(userdata={"pi_id": pi_id})
        client.on_connect = on_connect
        client.on_message = on_message
        clients[pi_id] = client

        client.connect(pi_id, 1883, RECONNECT_TIME)
        client.loop_start()

    update_thread = Thread(target=update_images, args=(image_sources,))
    update_thread.start()

    face_recognize_thread = Thread(target=get_face_recognize)
    face_recognize_thread.start()

    root.mainloop()

    # Wait for both threads to finish before exiting the program
    update_thread.join()
    face_recognize_thread.join()

    