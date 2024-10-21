import numpy as np
import cv2
import os

def calc_sad(left_patch, right_patch):
    return np.sum(np.abs(left_patch - right_patch))

def calc_ssd(left_patch, right_patch):
    return np.sum((left_patch - right_patch) ** 2)

def calc_ncc(left_patch, right_patch):
    left_mean = np.mean(left_patch)
    right_mean = np.mean(right_patch)
    numerator = np.sum((left_patch - left_mean) * (right_patch - right_mean))
    denominator = np.sqrt(np.sum((left_patch - left_mean) ** 2) * np.sum((right_patch - right_mean) ** 2))
    return numerator / (denominator + 1e-5)

def calculate_disparity(left_image, right_image, window_size, disparity_range, method, direction='left-to-right'):
    height, width = left_image.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)
    
    for y in range(window_size // 2, height - window_size // 2):
        for x in range(window_size // 2, width - window_size // 2):
            best_offset = 0
            min_value = float('inf') if method in ['ssd', 'sad'] else -float('inf')  
            left_patch = left_image[y - window_size // 2 : y + window_size // 2 + 1,
                                    x - window_size // 2 : x + window_size // 2 + 1]
            for offset in range(disparity_range):
                if direction == 'left-to-right' and x - offset - window_size // 2 >= 0:
                    right_patch = right_image[y - window_size // 2 : y + window_size // 2 + 1,
                                              x - offset - window_size // 2 : x - offset + window_size // 2 + 1]
                elif direction == 'right-to-left' and x + offset + window_size // 2 < width:
                    right_patch = right_image[y - window_size // 2 : y + window_size // 2 + 1,
                                              x + offset - window_size // 2 : x + offset + window_size // 2 + 1]
                else:
                    continue

                if method == 'ssd':
                    value = calc_ssd(left_patch, right_patch)
                    if value < min_value:
                        min_value = value
                        best_offset = offset
                elif method == 'sad':
                    value = calc_sad(left_patch, right_patch)
                    if value < min_value:
                        min_value = value
                        best_offset = offset
                elif method == 'ncc':
                    value = calc_ncc(left_patch, right_patch)
                    if value > min_value: 
                        min_value = value
                        best_offset = offset

            disparity_map[y, x] = best_offset

    disparity_map = (disparity_map / disparity_range) * 255
    return disparity_map.astype(np.uint8)

def apply_color_map_to_disparity(disparity_map):
    """ Apply a color map to the disparity map for better visualization. """
    return cv2.applyColorMap(disparity_map, cv2.COLORMAP_JET)

# Setup parameters
image_name = "triclopsi.jpg"
left_image_name = "triclopsi2l.jpg"
right_image_name = "triclopsi2r.jpg"
output_dir = "outputs"
left_image = cv2.imread(os.path.join("Assignment02 Disparity and Depth", left_image_name), cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(os.path.join("Assignment02 Disparity and Depth", right_image_name), cv2.IMREAD_GRAYSCALE)

window_size_list = [5, 10, 15]
disparity_ranges = [16]
methods = ["ssd", "sad", "ncc"]

for window_size in window_size_list:
    for disparity_range in disparity_ranges:
        for method in methods:
            for direction in ["left-to-right","right-to-left"]:
                print(f"Using window_size={window_size}, method={method}")
                
                # Calculate disparity map for left-to-right
                disparity_map_left_to_right = calculate_disparity(left_image, right_image, window_size, disparity_range, method, direction)
                
                # Apply color map to the disparity map
                colored_disparity_map = apply_color_map_to_disparity(disparity_map_left_to_right)
                
                # Save the colored disparity map
                output_filename = f'{image_name}_{direction}_{method}_{disparity_range}_{window_size}.png'
                cv2.imwrite(os.path.join(output_dir, output_filename), colored_disparity_map)
                print(f"Saved: {output_filename}")

            