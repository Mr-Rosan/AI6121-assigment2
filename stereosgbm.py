import os
import cv2
import numpy as np

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

def compute_disparity_and_sobel(left_image_path, right_image_path, output_dir, image_name="image", method="sgbm"):
    """
    先对输入图像应用 Sobel 滤波，然后计算左右视差图并保存结果。
    
    参数：
    - left_image_path: 左图像的路径
    - right_image_path: 右图像的路径
    - output_dir: 结果输出的目录
    - image_name: 图像名称的前缀，用于保存文件
    - method: 立体匹配的方法，默认为 "sgbm"
    """

    # 读取左右图像
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    if left_image is None or right_image is None:
        print("Error: Unable to load one of the images.")
        return

    # 对图像应用 Sobel 滤波
    sobel_left_image = apply_sobel_filter(left_image)
    sobel_right_image = apply_sobel_filter(right_image)

    # 保存 Sobel 处理后的图像
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_sobel_left.png"), sobel_left_image)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_sobel_right.png"), sobel_right_image)

    # 创建 StereoSGBM 对象
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16,  # 必须是16的倍数
        blockSize=5,
        P1=8 * 3 * 7 ** 2,
        P2=32 * 3 * 7 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 计算视差图
    disparity_map_left_to_right = stereo.compute(sobel_left_image, sobel_right_image).astype(np.float32) / 16.0
    disparity_map_right_to_left = stereo.compute(sobel_right_image, sobel_left_image).astype(np.float32) / 16.0

    # 归一化视差图以保存为图像
    disparity_map_left_to_right = cv2.normalize(disparity_map_left_to_right, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_left_to_right = np.uint8(disparity_map_left_to_right)

    disparity_map_right_to_left = cv2.normalize(disparity_map_right_to_left, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_right_to_left = np.uint8(disparity_map_right_to_left)

    disparity_map_left_to_right = cv2.applyColorMap(disparity_map_left_to_right, cv2.COLORMAP_JET)
    disparity_map_right_to_left = cv2.applyColorMap(disparity_map_right_to_left, cv2.COLORMAP_JET)
    # 保存视差图
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_disparity_map_left_to_right_{method}.png"), disparity_map_left_to_right)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_disparity_map_right_to_left_{method}.png"), disparity_map_right_to_left)

# 示例使用
left_image_name = "corridorl.jpg"
right_image_name = "corridorr.jpg"
output_dir = "outputs"
compute_disparity_and_sobel(
    left_image_path=os.path.join("Assignment02 Disparity and Depth", left_image_name),
    right_image_path=os.path.join("Assignment02 Disparity and Depth", right_image_name),
    output_dir=output_dir,
    image_name="corridor",
    method="sgbm"
)
