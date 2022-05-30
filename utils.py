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
    
    K = [674.9000244140625, 0.0, 632.750244140625, 0.0, 674.9000244140625, 352.6859436035156, 0.0, 0.0, 1.0]
    R = [1, -1, 0]
    t = [0.105, -0.580, 0.008]

    # build extrinsic matrix
    R = np.asarray(
        Rotation.from_euler('xyz', np.pi / 2 * np.array(R) ).as_matrix()
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