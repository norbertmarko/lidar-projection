import numpy as np


def calculate_matrices(file_path):
    """
    Calculates the intrinsic and extrinsic
    camera matrices for transformations and
    projections from KITTI calibration source 
    file.
    """
    with open(file_path) as file:
        lines = file.readlines()

    P_rect = []    
    for line in lines:
        
        title = line.strip().split(' ')[0]
        
        if len(title):
        
            if title[0] == "R":
                R_rect = np.array(line.strip().split(' ')[1: ], dtype=np.float32)
                R_rect = np.reshape(R_rect, (3, 3))
        
            elif title[0] == "P":
                p_r = np.array(line.strip().split(' ')[1: ], dtype=np.float32)
                p_r = np.reshape(p_r, (3, 4))
                P_rect.append(p_r)
        
            elif title[:-1] == "Tr_velo_to_cam":
                Tr = np.array(line.strip().split(' ')[1: ], dtype=np.float32)
                Tr = np.reshape(Tr, (3, 4))
                Tr = np.vstack([Tr,np.array([0, 0, 0, 1])])
    
    # R_rect, P_rect, Tr

    # Calculation
    R_cam2rect = np.hstack([R_rect, np.array([[0], [0], [0]])])
    R_cam2rect = np.vstack([R_cam2rect, np.array([0, 0, 0, 1])])
    
    P = np.matmul(P_rect[2], R_cam2rect)
    P = np.matmul(P, Tr)

    return P