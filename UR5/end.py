import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import socket
import time
from pose_utils import A_sim, A_real, V_sim, V_real, pose_str

# Constants
PORT = 30003
while True:
    mode = input("Select mode ('sim' or 'real'): ").strip().lower()

    if mode == "sim":
        HOST = "172.17.0.2"

        #Simulation Accelaration and Velocity

        A = A_sim
        V = V_sim

        break

    elif mode == "real":
        HOST = "192.168.0.153"  # <-- replace with your real robot IP

        A = A_real
        V = V_real

        break
    else:
        print("Invalid input. Please type 'sim' or 'real'.")


def main():
   
    pre_end = [-1.57, -1.57, -1.57, -1.57, 1.57, 0]
    pre_end_line = pose_str(pre_end)

   
    
    ur_script = (
        "def my_program():\n"
        f"  movej([{pre_end_line}], a={A}, v={V}, t=0, r=0)\n"
        "end\n"
        "my_program()\n"
    )

    print("Connecting to robot...")
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
