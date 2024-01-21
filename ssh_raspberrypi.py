import paramiko
import threading
import argparse
import time
import re
from queue import Queue

HOSTNAME_LEFT = 'raspberrypi-left'
HOSTNAME_MID = 'raspberrypi-mid'
HOSTNAME_RIGHT = 'raspberrypi-right'
USERNAME = 'pi'
PASSWORD = 'raspberrypi'

HOST = 'hieu-Inspiron-5570'
PYTHON_FOLDER_SOURCE = f'cd Desktop/face/ && source myenv/bin/activate'
FACE_EMBEDDING_COMMAND = f'python3 know_faces_embedding_client.py'

PYTHON_FOLDER_SOURCE = f'cd Desktop/face-recognize-flower/ && source myenv/bin/activate'
FACE_EMBEDDING_COMMAND = f'python3 know_faces_embedding_client.py --face_dataset=face_dataset/m/'
FACE_GET_COMMAND = f'python3 get_data_client.py'
FACE_TEST_COMMAND = f'python3 face_test_client.py'

#____________________________________ VARIABLES ___________________________
channels = []
threads = []
hosts = []

#________________________ FUNCTION ___________________________
def read_string(msg):
    while True:
        try:
            label = input(msg)
            if label is not None or len(label) != 0:
                return label
        except:
            print(f"ERROR: Not a string ({label})")
            continue

def extract_score_id(output):
    pattern = re.compile(r'(Score|ID): (\d+(\.\d+)?)')
    matches = pattern.findall(output)
    res = {key: value if key == 'ID' else float(value) for key, value, _ in matches}

    return res

def execute_code_pi(host: str, hostname: str, channel: paramiko.Channel, code: str, input_signal_cmd: Queue, output_signal: str, to_output: list[str], need_print: bool):
    try:
        channel.send(code)
        signal = ''
        cmd = ''

        while True:
            temp = ''
            if channel.recv_ready():
                temp = channel.recv(1024).decode('utf-8')

            if need_print:
                lines = temp.splitlines(keepends=True)
                for line in lines:
                    print(f'{host}: {line}', end='')

            if f'{args.username}@{hostname}' in temp:
                to_output.append((host, temp))
                break

            if len(output_signal) != 0 and output_signal in temp:
                to_output.append((host, temp))

            if not input_signal_cmd.empty():
                signal, cmd = input_signal_cmd.get()

            if len(signal) != 0:
                channel.send(cmd)
                signal = ''
            time.sleep(0.1)
        
    except Exception as e:
        print(f'Error on {host}: {e}')

def execute_code_pis(code: str, output_signal: str = '', to_output: list[str] = [], need_print: bool = False, block: bool = True):
    threads.clear()
    threads_queue = []
    for _, hostname, _, channel in channels:
        q = Queue()
        thread = threading.Thread(target=execute_code_pi, args=(host, hostname, channel, code, q, output_signal, to_output, need_print))
        thread.daemon = True
        thread.start()
        threads.append(thread)
        if not block:
            threads_queue.append(q)
    if block:
        for thread in threads: thread.join()
        return None
    else:
        return threads_queue
#________________________ START ___________________________
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="SSH face verify!")
    parser.add_argument(
        "--username",
        type=str,
        default=USERNAME,
        help=f"All the Raspberry Pi's login username! (deafault {USERNAME})",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=PASSWORD,
        help=f"All the Raspberry Pi's login password! (deafault {PASSWORD})",
    )
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
    parser.add_argument(
        "--know_faces",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help=f"If the know-faces has beed embeded then activate this! (default: False)",
    )
    args = parser.parse_args()

    # hosts.extend([(args.pi_left, HOSTNAME_LEFT), (args.pi_mid, HOSTNAME_MID), (args.pi_right, HOSTNAME_RIGHT)])
    hosts.append((HOST, HOST))
    print(hosts)
    try:
        for host, hostname in hosts:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host, username=args.username, password=args.password)
            channels.append((host, hostname, ssh, ssh.invoke_shell()))

        execute_code_pis('')
        execute_code_pis(PYTHON_FOLDER_SOURCE + '\r')

        if args.know_faces is False: execute_code_pis(FACE_EMBEDDING_COMMAND + '\r')

        if len(channels) != len(hosts):
            exit(f'Connected fail to some Pi!')

        outputs = []
        res = {
            'Score': [],
            'ID': [],
        }

        queues_input = execute_code_pis(FACE_TEST_COMMAND + '\r', 'ID', outputs, block=False)

        while True:
            res['ID'].clear()
            res['Score'].clear()
            outputs.clear()
            start = time.time()

            for q in queues_input:
                q.put(('Face_verify_input:', 's\r'))

            while len(outputs) != len(hosts):
                pass
            for output in outputs:
                _, text = output
                temp = extract_score_id(text)
                for key, val in temp.items():
                    res[key].append(val)

            print(res)
            if len(res['ID']) != len(hosts):
                print('Some camera cannot verify face!')
                break
            if len(set(res['ID'])) != 1:
                print('The face cannot be verify because the result of cameras is not match!')
                continue

            print(f'The face\'s ID is {res["ID"][0]} and the average score is {sum(res["Score"]) / len(res["Score"])}')
            print(f'Time: {time.time() - start}')
            
        # queues_input = execute_code_pis(FACE_GET_COMMAND + '\r', 'done', outputs, block=False)
        # while True:
        #     outputs.clear()
        #     label = read_string('Label: ')
        #     for q in queues_input:
        #         q.put(('Label:', label + '\r'))

        #     while len(outputs) != len(hosts):
        #         pass
        #     # print(outputs)
        #     if len(label) == 0:
        #         break
        #     label = ''
            

    except KeyboardInterrupt:
        print('Ending program!')
        for q in queues_input:
            q.put(('', 'exit'))
        for _, _, ssh, channel in channels:
            channel.close()
            ssh.close()
        for thread in threads: thread.join()
        exit()