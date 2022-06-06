import numpy as np
from scipy.spatial.transform import Rotation


def calculate_matrices(homogeneous: bool = True):
    """
    Calculates the intrinsic and extrinsic
    camera matrices for transformations and
    projections.

    ARGS
    ----
    `homogeneous (bool)`: If `True`, the function
        returns the matrices with homogeneous
        coordinates (default).

    RETURNS
    ------- 
    `Rt (np.array)`: The extrinsic matrix (4 x 4) containing
        the rotation matrix and translation vector.
    `K (np.array)`: The intrinsic matrix (3 x 4) containing
        the camera calibration parameters.
    `P (np.array)`: Projection matrix (3 x 4), the two above
        multiplied together.
    """
    
    # camera matrix
    K = [2787.812258, 0.000000, 881.039942, 0.000000, 2780.031855, 625.137965, 0.000000, 0.000000, 1.000000]
    
    # (rectified?) projection matrix
    K = [2742.022949, 0.000000, 875.146045, 0.000000, 2763.256592, 625.599612, 0.000000, 0.000000, 1.000000]
    
    # os1_lidar
    #R = [-1.576, 0.009, 1.536]
    #t = [-0.132, 0.039, -0.172]
    
    # os1_sensor
    t = [0.132, -0.039, -0.136]
    R = [-1.576, 0.009, -1.606]

    # build extrinsic matrix
    R = np.asarray(
        Rotation.from_euler('zyx', np.array(R) ).as_matrix()
    )

    t = np.array(t).reshape(3, 1)

    Rt = np.hstack([R, t])
    if homogeneous:
        Rt = np.vstack([Rt, np.array([0, 0, 0, 1])])

    # build intrinsic matrix
    K = np.array(K).reshape(3, 3)
    P = None    
    if homogeneous:
        K = np.hstack([np.array(K).reshape(3, 3), np.array([0, 0, 0]).reshape(3, 1)])
        P = np.dot(K, Rt)

    return Rt, K, P

if __name__ == '__main__':
    Rt, K, P = calculate_matrices(homogeneous=True)    
    print(Rt)