import paramiko
import threading
import argparse
import time
import re

HOSTNAME_LEFT = 'raspberrypi-left'
HOSTNAME_MID = 'raspberrypi-mid'
HOSTNAME_RIGHT = 'raspberrypi-right'
# HOSTS = ['hieu-Inspiron-5570']
USERNAME = 'pi'
PASSWORD = 'raspberrypi'
# USERNAME = 'hieu'
# PASSWORD = 'minhlemon03'
PYTHON_FOLDER_SOURCE = f'cd Desktop/face/ && source myenv/bin/activate'
# PYTHON_FOLDER_SOURCE = f'cd Desktop/face-recognize-flower/ && source myenv/bin/activate'
FACE_EMBEDDING_COMMAND = f'python3 know_faces_embedding_client.py'
# FACE_EMBEDDING_COMMAND = f'python3 know_faces_embedding_client.py --face_dataset=face_dataset/m/'
FACE_TEST_COMMAND = f'python3 face_test_client.py'

#____________________________________ VARIABLES ___________________________
channels = []
threads = []
hosts = []

#________________________ FUNCTION ___________________________
def extract_numeric_values(output):
    pattern = re.compile(r'(Score|ID): (\d+(\.\d+)?)')
    matches = pattern.findall(output)
    res = {key: value if key == 'ID' else float(value) for key, value, _ in matches}

    return res

def execute_code_pi(host: str, hostname: str, channel: paramiko.Channel, code: str, to_output: list[str], need_print: bool):
    try:
        channel.send(code)

        output = ""
        while True:
            output += channel.recv(1024).decode('utf-8')
            if f'{USERNAME}@{hostname}' in output:
                break
            time.sleep(1)

        if need_print:
            lines = output.splitlines()
            for line in lines:
                print(f'{host}: {line}')
        
        to_output.append(output)
    except Exception as e:
        print(f'Error on {host}: {e}')

def execute_code_pis(code: str, to_output: list[str] = [], need_print: bool = False):
    threads.clear()
    for _, hostname, _, channel in channels:
        thread = threading.Thread(target=execute_code_pi, args=(host, hostname, channel, code, to_output, need_print))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join()
#________________________ START ___________________________
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="SSH face verify!")
    parser.add_argument(
        "--pi_left",
        type=str,
        default=HOSTNAME_LEFT + '.local',
        help=f"The Raspberry Pi left's IP address or local hostname! (deafault {HOSTNAME_LEFT}.local)",
    )
    parser.add_argument(
        "--pi_mid",
        type=str,
        default=HOSTNAME_MID + '.local',
        help=f"The Raspberry Pi mid's IP address or local hostname! (deafault {HOSTNAME_MID}.local)",
    )
    parser.add_argument(
        "--pi_right",
        type=str,
        default=HOSTNAME_RIGHT + '.local',
        help=f"The Raspberry Pi right's IP address or local hostname! (deafault {HOSTNAME_RIGHT}.local)",
    )
    parser.add_argument(
        "--know_faces",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help=f"If the know-faces has beed embeded then activate this! (default: False)",
    )
    args = parser.parse_args()

    hosts.extend([(args.pi_left, HOSTNAME_LEFT), (args.pi_mid, HOSTNAME_MID), (args.pi_right, HOSTNAME_RIGHT)])
    print(hosts)
    try:
        for host, hostname in hosts:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host, username=USERNAME, password=PASSWORD)
            channels.append((host, hostname, ssh, ssh.invoke_shell()))

        execute_code_pis('')
        execute_code_pis(PYTHON_FOLDER_SOURCE + '\n', need_print=True)

        if args.know_faces is False: execute_code_pis(FACE_EMBEDDING_COMMAND + '\n', need_print=True)

        if len(channels) != 3:
            exit(f'Connected fail to some Pi!')

        while True:
            time.sleep(2)
            outputs = []
            res = {
                'Score': [],
                'ID': [],
            }
            execute_code_pis(FACE_TEST_COMMAND + '\n', outputs, need_print=True)
            for output in outputs:
                temp = extract_numeric_values(output)
                for key, val in temp.items():
                    res[key].append(val)

            print(res)
            if len(res['ID']) != len(hosts):
                print('Some camera cannot verify face!')
                continue
            if len(set(res['ID'])) != 1:
                print('The face cannot be verify because the result of cameras is not match!')
                continue

            print(f'The face\'s ID is {res["ID"][0]} and the average score is {sum(res["Score"]) / len(res["Score"])}')

    except KeyboardInterrupt:
        print('Ending program!')
        for thread in threads: thread.join()
        for _, _, ssh, channel in channels:
            channel.close()
            ssh.close()