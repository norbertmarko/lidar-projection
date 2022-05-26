import cv2
import numpy as np
from pyntcloud import PyntCloud

import utils, kitti_utils


def main(img_path, pc_path, pc_type, homogeneous=True, is_kitti=False):
	
	# read image
	img = cv2.imread(img_path)

	# read point cloud
	if pc_type == "bin":
		pc = np.fromfile(pc_path, dtype=np.float32)
		pc = np.reshape(pc, (-1, 4))
		pc = pc[:, :3].T

		if homogeneous & (pc.shape[0] < 4):
			pc = np.vstack([pc, np.ones(pc.shape[1])])

		# (3, ?) or (4, ?)
		print(f'[INFO] Point cloud starting shape: {pc.shape}')

	elif pc_type == "pcd":
		pc = PyntCloud.from_file(pc_path)

		x = np.array(pc.points['x'], dtype=np.float32)
		y = np.array(pc.points['y'], dtype=np.float32)
		z = np.array(pc.points['z'], dtype=np.float32)
		pc = np.vstack([x, y, z])

		if homogeneous & (pc.shape[0] < 4):
			pc = np.vstack([pc, np.ones(pc.shape[1])])

		# (3, ?) or (4, ?)
		print(f'[INFO] Point cloud starting shape: {pc.shape}')

	else:
		print("[ERROR] Provide a valid point cloud type!")
   
	# removing points behind image plane (approximate)
	# TODO! (lidar read func. & own projection, crop 3d visualize?)

	# calculate and read matrices
	if is_kitti:
		P = kitti_utils.calculate_matrices("assets/kitti_calib.txt")
	else:	
		Rt, K, P = utils.calculate_matrices(homogeneous)

	
	# transformations (extrensic and intrinsic)
	projected_points = np.matmul(P, pc).T

	#transformed_points = np.dot(Rt, pc)
	#projected_points = np.matmul(K, transformed_points).T

	temp = np.reshape( projected_points[:, 2], (-1, 1) )

	projected_points = projected_points[:, :3] / ( np.matmul( temp, np.ones([1, 3]) ) )


	# display projection on image
	depth_max = np.max(pc[:, 0])

	for (idx, i) in enumerate(projected_points):

		color = int((pc[0, idx] / depth_max) * 255)
		cv2.rectangle(img, (int(i[0] - 1), int(i[1] - 1)), (int(i[0] + 1), int(i[1] + 1)), (0, 0, color), -1)
	
	cv2.imshow("Projected Points", img)
	cv2.waitKey(0)


if __name__ == '__main__':
	
	img_path="assets/leaf_test.png"
	pc_path="assets/leaf_test.bin"
	
	main(img_path, pc_path, "bin", is_kitti=False)