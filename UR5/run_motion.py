import socket
import time 
import pandas as pd
from pose_utils import A_sim, A_real, V_sim, V_real, pose_str


def scanning():
    mode = input("Select mode ('sim' or 'real'): ").strip().lower()
    if mode == "sim":
        HOST = "172.17.0.2"
        A = A_sim
        V = V_sim
        SETTLE = 2   # short settle between moves; no real hardware to wait on

    elif mode == "real":
        HOST = "192.168.0.110"
        A = A_real
        V = V_real
        SETTLE = 2   
    else:
        print("Invalid mode. Exiting.")
        return

    # Secondary client interface (30002), NOT realtime (30003): pushing a
    # `def my_program() ... end` program to 30003 makes the controller suspend
    # its state broadcast on that port. 30002 runs the program instead.
    PORT = 30002

    try:
        df = pd.read_csv("path.csv")
        print(f"Loaded {len(df)} waypoints from 'path.csv'.")
    except FileNotFoundError:
        print("Error: 'path.csv' not found. Please run 'generate_path.py' first.")
        return

    lines = ["def scan_program():"]
    for i, row in df.iterrows():
        pose = [row['x'], row['y'], row['z'], row['rx'], row['ry'], row['rz']]
        pose_line = pose_str(pose)
        lines.append(f'  textmsg("waypoint {i + 1}/{len(df)} ({row["phase"]})")')
        lines.append(f"  movel(p[{pose_line}], a={A}, v={V}, t=0, r=0)")
        if row['phase'] == 'press':
            lines.append(f"  sleep({SETTLE})")
    lines.append("end\nscan_program()\n")
    ur_script = "\n".join(lines)

    print(f"Connecting to robot ({HOST}:{PORT})...")
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
    scanning()