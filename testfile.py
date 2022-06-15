import numpy as np
import cv2


h, w = 1024, 1280
dummy_img = np.full((h, w), 65535, dtype=np.uint16)

cv2.imshow("Dummy Image", dummy_img)
cv2.waitKey(0)

print(f"The data type of the image is {dummy_img.dtype}.")