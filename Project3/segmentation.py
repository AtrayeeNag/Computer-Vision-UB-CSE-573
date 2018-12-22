import cv2
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)

# Add padding to image
def add_padding(image, padding_width, background_value):

	h, w = image.shape

	padded_img = [[background_value for col in range(w + (padding_width*2))] for row in range(h + (padding_width*2))]

	for i in range(h):
		for j in range(w):
			padded_img[i + padding_width][j + padding_width] = image[i][j]

	return np.asarray(padded_img)


# Remove padding from image
def remove_padding(image, padding_width):

	h, w = image.shape

	no_padded_img = [[0 for col in range(w - (padding_width*2))] for row in range(h - (padding_width*2))]

	for i in range(padding_width, h-padding_width):
		for j in range(padding_width, w-padding_width):
			no_padded_img[i - padding_width][j - padding_width] = image[i][j]

	return np.asarray(no_padded_img)


# Apply masking with given kernel
def apply_masking(mask, img, kernel_radius):

	padding_width = int((len(mask[0])-1)/2)
	background_value =0
	img = add_padding(img, padding_width, background_value)

	masked_img = np.zeros(img.shape)
	height, width = img.shape

	for x in range(kernel_radius, height-kernel_radius):
	    for y in range(kernel_radius, width-kernel_radius):

	        loop_end = (kernel_radius*2)+1
	        sum = 0

	        for i in range(0,loop_end):
	            for j in range(0,loop_end):

	                sum += mask[i][j] * img[x-kernel_radius+i][y-kernel_radius+j]

	        masked_img[x][y] = sum

	return remove_padding(masked_img, padding_width)


# Counting intensities
def plot_histogram(img):

    intensity_count = np.zeros(256)
    height, width = img.shape

    for i in range(0, height):
        for j in range(0, width):

            intensity_count[img[i][j]] += 1

    intensity_count[0] = 0

    plt.plot([col for col in range(256)], intensity_count)
    plt.show()


#Thresholding
def threshold_img(img, threshold):

    height, width = img.shape
    for i in range(0, height):
        for j in range(0, width):

            if img[i][j] < threshold:
                img[i][j] = 0
            else:
                img[i][j] = 255
    return img


# Heuristic way to find optimal threshold
def optimal_thresholding(img):

    t_init = 230
    while True:

        height, width = img.shape
        right_sum, left_sum, right_count, left_count = 0, 0, 0, 0

        for i in range(0, height):
            for j in range(0, width):

                if img[i][j] < 190:
                    continue
                elif img[i][j] < t_init:
                    left_sum += img[i][j]
                    left_count += 1
                else:
                    right_sum += img[i][j]
                    right_count += 1

        t_next = 0.5 * ((left_sum/left_count) + (right_sum/right_count))

        if t_init == t_next:
            break
        t_init = t_next

    return int(t_init)


# Main

# Question 2a)
img_point = cv2.imread("original_imgs/turbine-blade.jpg", 0)

mask = [[-1,-1,-1],
         [-1, 8,-1],
         [-1,-1,-1]]

masked_img = apply_masking(mask, img_point, 1)

masked_img = masked_img*255/np.max(masked_img)

masked_img = threshold_img(masked_img, 120)

cv2.imwrite("output_imgs/masked.jpg",masked_img)


# Question 2b)
img = cv2.imread("original_imgs/segment.jpg", 0)

plot_histogram(img)

opt_threshold = optimal_thresholding(img)

img = threshold_img(img, opt_threshold)

img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

# Plot boundary Box
cv2.rectangle(img, (381, 37), (429, 258), (0,0,255), 2)
cv2.rectangle(img, (331, 19), (370, 295), (0,0,255), 2)
cv2.rectangle(img, (245, 72), (307, 211), (0,0,255), 2)
cv2.rectangle(img, (158, 122), (207, 170), (0,0,255), 2)

cv2.imwrite("output_imgs/segment_op.jpg", img)
