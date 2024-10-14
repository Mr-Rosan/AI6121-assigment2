import os
import cv2
import numpy as np

left_image_name = "corridorl.jpg"
right_image_name = "corridorr.jpg"
output_dir = "outputs"
left_image = cv2.imread(os.path.join("Assignment02 Disparity and Depth",left_image_name), cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(os.path.join("Assignment02 Disparity and Depth",right_image_name),cv2.IMREAD_GRAYSCALE)

# 创建 StereoBM 对象
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

image_name = "corridor"
method = "bm"
# window_size_list = [5, 10, 15]
# disparity_ranges = [16]

disparity_map_left_to_right = stereo.compute(left_image, right_image)
disparity_map_right_to_left = stereo.compute(right_image, left_image)

disparity_map_left_to_right = cv2.normalize(disparity_map_left_to_right, None, 0, 255, cv2.NORM_MINMAX)
disparity_map_left_to_right = np.uint8(disparity_map_left_to_right)

disparity_map_right_to_left = cv2.normalize(disparity_map_right_to_left, None, 0, 255, cv2.NORM_MINMAX)
disparity_map_right_to_left = np.uint8(disparity_map_right_to_left)

cv2.imwrite(os.path.join(output_dir,image_name + ' disparity_map_left_to_right' + method + '.png'),disparity_map_left_to_right)
cv2.imwrite(os.path.join(output_dir,image_name + ' disparity_map_right_to_left' + method + '.png'),disparity_map_right_to_left)

sobelx = cv2.Sobel(disparity_map_left_to_right, cv2.CV_64F, 1, 0, ksize=5)  # X方向上的梯度
sobely = cv2.Sobel(disparity_map_left_to_right, cv2.CV_64F, 0, 1, ksize=5)  # Y方向上的梯度

sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
sobel_combined = np.uint8(sobel_combined)
cv2.imwrite(os.path.join(output_dir,image_name + ' disparity_map_left_to_right_' + method + '_after_sobel.png'),sobel_combined)

sobelx = cv2.Sobel(disparity_map_right_to_left, cv2.CV_64F, 1, 0, ksize=5)  # X方向上的梯度
sobely = cv2.Sobel(disparity_map_right_to_left, cv2.CV_64F, 0, 1, ksize=5)  # Y方向上的梯度

sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
sobel_combined = np.uint8(sobel_combined)
cv2.imwrite(os.path.join(output_dir,image_name + ' disparity_map_right_to_left_' + method + '_after_sobel.png'),sobel_combined)

