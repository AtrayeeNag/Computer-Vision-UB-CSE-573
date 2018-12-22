import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)

# Add padding around image
def add_padding(image, padding_width, background_value):

	h, w = image.shape

	padded_img = [[background_value for col in range(w + (padding_width*2))] for row in range(h + (padding_width*2))]

	for i in range(h):
		for j in range(w):
			padded_img[i + padding_width][j + padding_width] = image[i][j]


	return np.asarray(padded_img)


# Remove padding around img
def remove_padding(image, padding_width):

	h, w = image.shape

	no_padded_img = [[0 for col in range(w - (padding_width*2))] for row in range(h - (padding_width*2))]

	for i in range(padding_width, h-padding_width):
		for j in range(padding_width, w-padding_width):
			no_padded_img[i - padding_width][j - padding_width] = image[i][j]


	return np.asarray(no_padded_img)


# Apply masking with a kernel
def apply_masking(mask, img, kernel_radius):

	padding_width = int((len(mask[0])-1)/2)
	background_value =220
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


def gaussian(x, mu, sigma):
  return math.exp( -(((x-mu)/(sigma))**2)/2.0 )


def get_gaussian_kernel(sigma):
    kernel_radius = 3
    # compute the kernel elements
    hkernel = [gaussian(x, kernel_radius, sigma) for x in range(2*kernel_radius+1)]
    vkernel = [x for x in hkernel]
    kernel2d = [[xh*xv for xh in hkernel] for xv in vkernel]
    # normalize the kernel elements
    kernelsum = sum([sum(row) for row in kernel2d])
    kernel2d = [[x/kernelsum for x in row] for row in kernel2d]
    return kernel2d


def apply_gaussian_blur(img, sigma):
    height, width = img.shape
    blur_image = apply_masking(get_gaussian_kernel(sigma), img, 3)
    return blur_image


# Normalize image by dividing absolute value of pixels with the max value
def normalize_edge(edge, threshold):
    height, width = edge.shape
    max_val=0
    for i in range(0, height):
        for j in range(0, width):
            edge[i][j] = abs(edge[i][j])
            max_val = max(max_val,edge[i][j])
    for i in range(0, height):
        for j in range(0, width):
            norm = (edge[i][j]/max_val)*255
            if norm > threshold:
                edge[i][j] = 255
            else:
                edge[i][j] = 0
    return edge


# Detect edges of the image
def detect_edge(img):

    edge_x = np.zeros(img.shape)
    edge_y = np.zeros(img.shape)

    # Detect edge along x-direction
    sobel_x = [[-1,0,1],
               [-2,0,2],
               [-1,0,1]]

    edge_x = apply_masking(sobel_x, img, 1)
    edge_x = normalize_edge(edge_x, 20)

	# Detect edge along x-direction
    sobel_y = [[-1,-2,-1],
               [0,0,0],
               [1,2,1]]
    edge_y = apply_masking(sobel_y, img, 1)
    edge_y = normalize_edge(edge_y, 48)

    return edge_x, edge_y


# Find Hough accumulator for Lines: voting
def find_hough_acc_lines(img):

    height, width = img.shape
    p_range = int(math.sqrt(height ** 2 + width ** 2))
    accumulator = np.zeros((180, p_range*2))

    for i in range(0, height):
        for j in range(0, width):
            if img[i][j] ==0:
                continue
            else:
                for k in range(-90, 90):
                    theta = math.radians(k)
                    p = int((j * math.cos(theta)) + (i * math.sin(theta)))
                    accumulator[k+90][p+p_range] += 1

    return accumulator


# Find the pixels with the highest intensities
def find_hough_peaks(accumulator, threshold):

    height, width = accumulator.shape
    peak = np.unravel_index(np.argsort(accumulator.ravel())[-threshold:], (height, width))
    return peak


# Detect vertical and slanting lines with the peaks
def find_hough_lines(img, peak):

    img_vert = np.copy(img)
    img_slant = np.copy(img)
    height, width, _ = img.shape
    theta_arr = peak[0]
    p_range = int(math.sqrt(height ** 2 + width ** 2))
    p_arr = np.subtract(peak[1], p_range)

    for z in range(0, theta_arr.shape[0]):

        theta = theta_arr[z]-90
        theta_rad = math.radians(theta)

        if theta != 0 and theta == -2:

            for x in range(height):
                y = int((p_arr[z] - (x * math.sin(theta_rad)))/math.cos(theta_rad))
                if y<width and y>-1:
                    img_vert[x][y] = (0,255,0)

        elif theta != 0 and (theta == -36):

            for x in range(height):
                y = int((p_arr[z] - (x * math.sin(theta_rad)))/math.cos(theta_rad))
                if y<width and y>-1:
                    img_slant[x][y] = (0,255,0)

    return img_vert, img_slant


# Find Hough accumulator for Circle: voting
def find_hough_acc_circle(img):

    h, w = img.shape
    r_range = int(math.sqrt(h ** 2 + w ** 2))
    accumulator = np.zeros(img.shape)
    R = 20

    for i in range(0, h):
        for j in range(0, w):
            if img[i][j] ==0:
                continue
            else:
                for theta in range(0, 360):
                    theta_rad = math.radians(theta)
                    a = int(i - round((R * math.cos(theta_rad))))
                    b = int(j - round((R * math.sin(theta_rad))))
                    if a<h and a>-1 and b<w and b>-1:
                        accumulator[a][b] += 1
    return accumulator


# Detect vertical and slanting lines with the peaks
def find_hough_circle(img, peak):
    img_circle = np.copy(img)
    height, width, _ = img.shape
    a_arr = peak[0]
    b_arr = peak[1]
    R = 20

    for z in range(a_arr.shape[0]):

        for theta in range(360):
            theta_rad = math.radians(theta)
            x = int(a_arr[z] + round((R * math.cos(theta_rad))))
            y = int(b_arr[z] + round((R * math.sin(theta_rad))))
            if y<width and y>-1 and x<height and x>-1:
                img_circle[x][y] = (0,255,0)

    return img_circle


# Main
original_img = cv2.imread("original_imgs/hough.jpg", 0)
original_img_color = cv2.imread("original_imgs/hough.jpg")

blurred_img = apply_gaussian_blur(original_img, math.sqrt(2))

edge_x_img, edge_y_img = detect_edge(blurred_img)

# Detect lines in an image
accumulator_line = find_hough_acc_lines(edge_x_img)
cv2.imwrite("output_imgs/accumulator_line.jpg", accumulator_line)

peak_lines = find_hough_peaks(accumulator_line, 2000)

img_vert, img_slant = find_hough_lines(original_img_color, peak_lines)
cv2.imwrite("output_imgs/red_line.jpg", img_vert)
cv2.imwrite("output_imgs/blue_lines.jpg", img_slant)

# Detect circles in an image
accumulator_circle = find_hough_acc_circle(edge_y_img)
cv2.imwrite("output_imgs/accumulator_circle.jpg", accumulator_circle)

peak_circle = find_hough_peaks(accumulator_circle, 550)

img_circle = find_hough_circle(original_img_color, peak_circle)
cv2.imwrite("output_imgs/coin.jpg", img_circle)
