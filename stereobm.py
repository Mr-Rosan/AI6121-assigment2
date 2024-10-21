import os
import cv2
import numpy as np

def compute_disparity_and_sobel(left_image_path, right_image_path, output_dir, 
                                image_name="image", method="bm", num_disparities=16, block_size=15):
    """
    计算左右视差图并应用Sobel算子处理，生成并保存结果。
    
    参数：
    - left_image_path: 左图像的路径
    - right_image_path: 右图像的路径
    - output_dir: 结果输出的目录
    - image_name: 图像名称的前缀，用于保存文件
    - method: 用于命名的字符串，默认值为 "bm"
    - num_disparities: 视差的最大范围，默认值为 16
    - block_size: 匹配块的大小，默认值为 15
    """

    # 读取左右图像
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    if left_image is None or right_image is None:
        print("Error: Unable to load one of the images.")
        return
    
    def apply_sobel_filter(image):
        """
        对图像应用 Sobel 滤波，返回滤波后的图像。
        """
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # X方向上的梯度
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Y方向上的梯度

        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
        sobel_combined = np.uint8(sobel_combined)
    
        return sobel_combined

    sobel_left_image = apply_sobel_filter(left_image)
    sobel_right_image = apply_sobel_filter(right_image)

    cv2.imwrite(os.path.join(output_dir, f"{image_name}_sobel_left.png"), sobel_left_image)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_sobel_right.png"), sobel_right_image)

    # 创建 StereoBM 对象
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    # 计算视差图
    disparity_map_left_to_right = stereo.compute(sobel_left_image, sobel_right_image)
    disparity_map_right_to_left = stereo.compute(sobel_right_image, sobel_left_image)

    # 归一化视差图以保存为图像
    disparity_map_left_to_right = cv2.normalize(disparity_map_left_to_right, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_left_to_right = np.uint8(disparity_map_left_to_right)

    disparity_map_right_to_left = cv2.normalize(disparity_map_right_to_left, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_right_to_left = np.uint8(disparity_map_right_to_left)

    disparity_map_left_to_right = cv2.applyColorMap(disparity_map_left_to_right, cv2.COLORMAP_JET)
    disparity_map_right_to_left = cv2.applyColorMap(disparity_map_right_to_left, cv2.COLORMAP_JET)

    # 保存视差图
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_disparity_map_left_to_right_{method}.png"), disparity_map_left_to_right)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_disparity_map_right_to_left_{method}.png"), disparity_map_right_to_left)

    # 对左右视差图分别应用Sobel算子
    
# 示例使用
left_image_name = "triclopsi2l.jpg"
right_image_name = "triclopsi2r.jpg"
output_dir = "outputs"
compute_disparity_and_sobel(
    left_image_path=os.path.join("Assignment02 Disparity and Depth", left_image_name),
    right_image_path=os.path.join("Assignment02 Disparity and Depth", right_image_name),
    output_dir=output_dir,
    image_name="triclopsi2",
    method="bm",
    num_disparities=16,
    block_size=15
)
