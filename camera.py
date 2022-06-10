import numpy as np
import scipy.linalg as linalg
import scipy.spatial.transform as transform
import cv2


class Camera:
	"""Class for representing pinhole cameras.
	Initializes P = K[R|t] camera model."""

	def __init__(self, P=None):
		self.P = P
		self.K = None # intrinsic matrix (calibration matrix)
		self.R = None # rotation (pose)
		self.t = None # translation (position)
		self.Rt = None # extrinsic matrix
		self.c = None # camera center


	def project(self, X):
		"""
		Project points in X (4 * n array) and normalize coordinates.
		This is equivalent to a rotation and translation:
			
			rotated_points = np.dot(Rt, pc)  // (3D World Frame -> 3D Camera Frame)
		
			The rotated points are sliced into x, y, z than pixel values
			are calculated and normalized. // (3D Camera Frame -> 2D Image Frame)

			u = (K[0, 0] * x) / z + K[0, 2] // matrix elements: fx, cx
			v = (K[1, 1] * y) / z + K[1, 2] // matrix elements: fy, cy
		"""
		
		x = np.dot(self.P, X)
		
		for i in range(3):
			x[i] /= x[2]
		
		return x


	def deproject(self):
		"""
		Deproject an (x, y) point, assuming flat surface.
		"""
		pass

	
	def compose(self, K, R, t):
		"""Compose projection matrix P from K, R and t."""

		# rotation matrix
		self.R = np.asarray(
			transform.Rotation.from_euler('xyz', np.pi / 2 * np.array(R) ).as_matrix()
		)
		
		# translation vector
		self.t = np.array(t).reshape(3, 1)

		# extrinsic matrix
		self.Rt = np.hstack([self.R, self.t])
		self.Rt = np.vstack([self.Rt, np.array([0, 0, 0, 1])])

		# intrinsic matrix
		self.K = np.hstack([np.array(K).reshape(3, 3), np.array([0, 0, 0]).reshape(3, 1)])

		# projection matrix
		self.P = np.dot(self.K, self.Rt)


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


	def center(self):
		""" Compute and return the camera center, given projection matrix P.""" 
		
		if self.c is not None: 
			return self.c 
		else:
		# compute c by factoring 
			self.factor() 
			self.c = -np.dot(self.R.T, self.t) 
		
		return self.c


	def visualize_projection(self, x, img_shape=(256, 512)):
		"""
		Visualize projected points on a black image
		with the shape of the camera frame. // (n * 3 array) -> (3 * n array)
		"""
		dummy_img = np.zeros_like(img_shape)
		x = x.T

		# point attributes
		radius = 2
		color = (0, 0, 255)
		thickness = -1

		for i in range(len(x)):
			coords = (int(x[0, i]), int(x[1, i]))
			cv2.circle(dummy_img, coords, radius, color, thickness)

		cv2.imshow("Projected Points", dummy_img)
		cv2.waitKey(0)


if __name__ == '__main__':
	cam = Camera()