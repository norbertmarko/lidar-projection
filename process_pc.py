import cv2
import imutils
import numpy as np


# TODO: fix color scaling by depth
def crop_projected_pc(proj_pc_path):

    proj_pc = cv2.imread(proj_pc_path)
    proj_pc = proj_pc[45:530, :]

    return proj_pc


def apply_bilateral_filter(proj_pc):
    #proj_pc = cv2.cvtColor(proj_pc, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(proj_pc, d=30, sigmaColor=1000, sigmaSpace=1000)
    return bilateral


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def main():
    print("[INFO] Code started!")
    proj_pc = crop_projected_pc("outputs/projected_raw/proj_pc_raw.png")

    #proj_pc = apply_bilateral_filter(proj_pc)

    #proj_pc = increase_brightness(proj_pc)
    
    cv2.imshow("Cropped Projected PC - Raw", proj_pc)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
