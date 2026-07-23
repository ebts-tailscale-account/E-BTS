import pandas as pd
import numpy as np
from pose_utils import contact_to_tcp_position, FIXED_ROTVEC, approach_height, contact_depth

def generate_diode_matrix():
    print(" Generating path for testing the sensor ")
    
    x_corner = 0.01042 
    y_corner = -0.49942
    z_surface = 0.223 # when the tip touch the sensor.
#change to our first point use calibration to find the corner of the sensor

    step_x = 0.006 #4points in x direction
    step_y = 0.007 #4points in y direction 
    rows = 4
    cols = 4

    waypoints = []
    
    for i in range (rows):
        for j in range (cols):

            contact_xyz = [x_corner - (i * step_x), y_corner - (j * step_y), z_surface]

            z_hover = z_surface + approach_height
            z_press = z_surface - contact_depth
            z_retract = z_hover
           
            for (z, phase) in [(z_hover, 'hover'), (z_press, 'press'), (z_retract, 'retract')]:
                tcp = contact_to_tcp_position([contact_xyz[0], contact_xyz[1], z])
                pose = np.concatenate([tcp, FIXED_ROTVEC])
                waypoints.append({
                                    'x': pose[0],
                                    'y': pose[1],
                                    'z': pose[2],
                                    'rx': pose[3],
                                    'ry': pose[4],
                                    'rz': pose[5],
                                    'phase': phase
                                })
    df = pd.DataFrame(waypoints)
    df.to_csv("path.csv", index=False)

    print(" Data points exported to file 'path.csv'!")

if __name__ == "__main__":
    generate_diode_matrix()