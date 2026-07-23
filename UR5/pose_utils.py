import numpy as np
from scipy.spatial.transform import Rotation as R

TOOL_TIP_OFFSET = np.array([0.0,0.0,0.0])


FIXED_ROTVEC = np.array([0.0,-3.14,0.0])

A_sim = 0.5
V_sim = 0.5
A_real = 0.1
V_real = 0.05

approach_height = 0.01 #1cm
contact_depth = 0.002  #2mm

def pose_str(p):
    return ",".join([f"{x:.6f}" for x in p])

def rotvec_to_matrix(rotvec):
    """Chuyển đổi Vector xoay của Robot UR thành Ma trận xoay 3x3 để nhân toán học"""
    return R.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()

def contact_to_tcp_position(contact_xyz, rotvec=FIXED_ROTVEC, tip_offset=None):
   
    if tip_offset is None:
        tip_offset = TOOL_TIP_OFFSET   
        
    contact_xyz = np.asarray(contact_xyz, dtype=float)
    tip_offset = np.asarray(tip_offset, dtype=float)
    
    rotation_matrix = rotvec_to_matrix(rotvec)
    
    tcp_position = contact_xyz - rotation_matrix @ tip_offset
    
    return tcp_position