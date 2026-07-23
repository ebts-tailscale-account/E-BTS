import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import socket
import time
from pose_utils import pose_str, A_sim, A_real, V_sim, V_real


def main():
    mode = input("Select mode ('sim' or 'real'): ").strip().lower()
    if mode == "sim":
        HOST = "172.17.0.2"
        A = A_sim
        V = V_sim
    elif mode == "real":
        HOST = "192.168.0.153"
        A = A_real
        V = V_real
    else:
        print("Invalid mode. Exiting.")
        return

    PORT = 30003

    home_pose = [0, -1.57, 0, -1.57, 0 , 0]
    home_pose_line = pose_str(home_pose)

    pre_home_pose = [-1.57, -1.57, 0, -1.57, 0 , 0]
    pre_home_pose_line = pose_str(pre_home_pose)

    ur_script = (
        "def my_program():\n"
        f"  movej([{pre_home_pose_line}], a={A}, v={V}, t=0, r=0)\n"
        f"  movej([{home_pose_line}], a={A}, v={V}, t=0, r=0)\n"
        "end\n"
        "my_program()\n"
    )
    print("Prehome -> Home position")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        s.sendall(ur_script.encode('ascii'))
        time.sleep(1)
        s.close()
        print("Script sent successfully! The robot should be moving.")
    except Exception as e:
        print(f"Failed to connect to the robot: {e}")

    
if __name__ == "__main__":
    main()
