import cv2
import numpy as np

# Load and resize images
bg1_img = cv2.imread('./Exercise04_Data/GreenBackground.png', 1)
bg1_img = cv2.resize(bg1_img, (678, 381))

bg2_img = cv2.imread('./Exercise04_Data/NewBackground.jpg', 1)
bg2_img = cv2.resize(bg2_img, (678, 381))

obj = cv2.imread('./Exercise04_Data/Object.png', 1)
obj = cv2.resize(obj, (678, 381))

def compute_difference(bg_img, input_img):
    """
    Compute the difference between the background image and the input image.
    Convert the difference to a single channel image.
    """
    difference_three_channel = cv2.absdiff(bg_img, input_img)
    difference_single_channel = np.sum(difference_three_channel, axis=2) / 3.0
    difference_single_channel = difference_single_channel.astype('uint8')
    return difference_single_channel

def compute_binary_mask(difference_single_channel):
    """
    Compute a binary mask from the single channel difference image.
    """
    difference_binary = np.where(difference_single_channel >= 75, 255, 0)
    difference_binary = np.stack((difference_binary,) * 3, axis=-1).astype('uint8')
    return difference_binary

def replace_background(bg1_image, bg2_image, ob_image):
    """
    Replace the background of the object image with a new background.
    """
    difference_single_channel = compute_difference(bg1_image, ob_image)
    binary_mask = compute_binary_mask(difference_single_channel)
    output = np.where(binary_mask == 255, ob_image, bg2_image)
    return output

# Replace the background and display the output
if __name__ == "__main__":
    output = replace_background(bg1_img, bg2_img, obj)
    cv2.imshow('Object', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
