import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt


class Camera:
    """Class for representing pinhole cameras.
    Initializes P = K[R|t] camera model."""

    def __init__(self, P):
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation
        self.c = None # camera center

    def project(self, X):
        """Project points in X (4 * n array) and normalize coordinates."""
        
        x = np.dot(self.P, X)
        
        for i in range(3):
            x[i] /= x[2]
        
        return x

    def factor(self):
        """Factorize the camera matrix into K, R, t as P = K[R|t]."""

        # factor first 3*3 part
        K, R = linalg.rq(self.P[:, :3])

        # make diagonal of K positive
        T = np.diag( np.sign(np.diag(K)) )

        if linalg.det(T) < 0:
            T[1, 1] *= -1

        self.K = np.dot(K, T)
        self.R = np.dot(T, R) # T is its own inverse
        self.t = np.dot(linalg.inv(self.K), self.P[:, 3])

        return self.K, self.R, self.t


def rotation_matrix(a):
    """ Creates a 3D rotation matrix for rotation around the axis of the vector a. """ 
    R = np.eye(4)
    R[:3, :3] = linalg.expm(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]
    ) 
    return R

     
def project_house():
    points = np.loadtxt('assets/house.p3d').T
    
    points = np.vstack(
        (points, np.ones(points.shape[1]))
    )

    # setup camera
    P = np.hstack(
        (np.eye(3), np.array([[0], [0], [-10]]))
    )
    cam = Camera(P)
    x = cam.project(points)

    # plot projection
    plt.figure()
    plt.plot(x[0], x[1], 'k.')
    plt.show()

    # create transformation 
    r = 0.05 * np.random.rand(3) 
    rot = rotation_matrix(r)

    # # rotate camera and project 
    # plt.figure() 
    # for t in range(20): 
    #     cam.P = np.dot(cam.P,rot) 
    #     x = cam.project(points) 
    #     plt.plot(x[0],x[1],'k.') 
    #     plt.show()


if __name__ == '__main__':
    
    # house point cloud projection
    project_house()