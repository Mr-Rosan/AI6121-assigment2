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
    return numerator / (denominator + 1e-5)  # 加上一个小数防止除0


def calculate_disparity(left_image, right_image, window_size, disparity_range, method, direction='left-to-right'):
    height, width = left_image.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)
    print("Begin calculate_disparity_map")
    
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


# left_image = cv2.imread('./Assignment02 Disparity and Depth/corridorl.jpg', cv2.IMREAD_GRAYSCALE)
# right_image = cv2.imread('./Assignment02 Disparity and Depth/corridorr.jpg', cv2.IMREAD_GRAYSCALE)


# window_size = 5
# disparity_range = 16
# matching_method = 'ssd'  

# disparity_map_left_to_right = calculate_disparity(left_image, right_image, window_size, disparity_range, matching_method, 'left-to-right')


# disparity_map_right_to_left = calculate_disparity(right_image, left_image, window_size, disparity_range, matching_method, 'right-to-left')


# cv2.imwrite('disparity_map_left_to_right.png', disparity_map_left_to_right)
# cv2.imwrite('disparity_map_right_to_left.png', disparity_map_right_to_left)


# cv2.imshow('Left to Right Disparity Map', disparity_map_left_to_right)
# cv2.imshow('Right to Left Disparity Map', disparity_map_right_to_left)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
image_name = "corridor"
left_image_name = "corridorl.jpg"
right_image_name = "corridorr.jpg"
output_dir = "outputs"
left_image = cv2.imread(os.path.join("Assignment02 Disparity and Depth",left_image_name), cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(os.path.join("Assignment02 Disparity and Depth",right_image_name),cv2.IMREAD_GRAYSCALE)
window_size_list = [5, 10, 15]
disparity_ranges = [16]
methods = ["ssd", "sad", "ncc"]
for window_size in window_size_list:
    for disparity_range in  disparity_ranges:
        for method in methods:
            print(f"Using {window_size} and {method}")
            disparity_map_left_to_right = calculate_disparity(left_image, right_image, window_size, disparity_range, method, 'left-to-right')
            disparity_map_right_to_left = calculate_disparity(right_image, left_image, window_size, disparity_range, method, 'right-to-left')
            print("Now writing images")
            cv2.imwrite(os.path.join(output_dir,image_name + ' disparity_map_left_to_right' + " " + method + " " +str(disparity_range) + " " +str(window_size) +".png"),disparity_map_left_to_right)
            print(os.path.join(output_dir,image_name,'disparity_map_left_to_right' + method + str(disparity_range) + str(window_size) +".png"))
            cv2.imwrite(os.path.join(output_dir,image_name + ' disparity_map_right_to_left' + " " + method + " " + str(disparity_range) + " " +str(window_size) +".png"),disparity_map_right_to_left)



