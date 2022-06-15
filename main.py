import cv2
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt

import utils, kitti_utils, rellis_utils


def save_img(output_path, img, is_grayscale=True):
    """Saves image in 16-bit (unsigned) format."""
    img = (img).astype(np.uint16)
    if is_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, img)
    print("[INFO] Image saved successfully!")


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
       
        #filter by 'x'
        pc.points = pc.points[pc.points["x"]>=0]

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
   
    # TODO: removing points behind image plane (approximate) - bin version!
    # TODO! (lidar read func. & own projection, crop 3d visualize?)

    # calculate and read matrices
    if is_kitti:
        P = kitti_utils.calculate_matrices("assets/kitti_calib.txt")
    else:	
        Rt, K, P = utils.calculate_matrices(homogeneous)

    
    # transformations (extrensic and intrinsic)
    projected_points = np.dot(P, pc).T

    #transformed_points = np.dot(Rt, pc)
    #projected_points = np.dot(K, transformed_points).T


    # temp = np.reshape( projected_points[:, 2], (-1, 1) )

    # projected_points = projected_points[:, :3] / ( np.matmul( temp, np.ones([1, 3]) ) )

    # TODO: what coordinate to normalize with? why do it at all?
    for i in range(3):
         projected_points[:, i] /= projected_points[:, 2]

    # create dummy image to display the point cloud on it
    dummy_img = np.full_like(img, 0, dtype=np.uint16)

    # display projection on image
    depth_max = np.max(pc[0, :])
    # radius of projected points
    radius = 0

    # subtract color value from the maximum value so the distances closer appear brighter
    for (idx, i) in enumerate(projected_points):
        color = int( (pc[0, idx] / depth_max) * 65536)
        cv2.circle(dummy_img, ( int(i[0]), int(i[1]) ), radius, (0, 0, 65536 - color), -1)
    
        
        #cv2.rectangle(dummy_img, (int(i[0] - 1), int(i[1] - 1)), (int(i[0] + 1), int(i[1] + 1)), 65536 - color, -1)
    
    cv2.imshow("Projected Points", dummy_img)
    cv2.waitKey(0)
    # cv2.imwrite("outputs/projected_raw/proj_pc_raw.png", dummy_img)
    save_img("outputs/projected_raw/proj_pc_raw.png", dummy_img)

if __name__ == '__main__':
    
    img_path="assets/leaf_test.png"
    pc_path="assets/leaf_test.pcd"
    
    main(img_path, pc_path, "pcd", is_kitti=False)

    
    #TODO: 1. save it as uint16 properly
    


    
    #TODO: 2. use smaller points for point cloud (make it close to kitti example projections)
    #TODO: 3. scale depth properly (check with example bin - print values - think trough calc.)
