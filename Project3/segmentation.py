import cv2
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)


def apply_masking(sobel, img, kernel_radius):

    masked_img = np.zeros(img.shape)
    height, width = img.shape

    for x in range(kernel_radius, height-kernel_radius):
        for y in range(kernel_radius, width-kernel_radius):

            loop_end = (kernel_radius*2)+1
            sum = 0

            for i in range(0,loop_end):
                for j in range(0,loop_end):

                    sum += sobel[i][j] * img[x-kernel_radius+i][y-kernel_radius+j]

            masked_img[x][y] = sum

    return masked_img


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

    return img


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


# img = cv2.imread("point.jpg", 0)
# sobel = [[-1,-1,-1],
#          [-1, 8,-1],
#          [-1,-1,-1]]
# masked_img = apply_masking(sobel, img, 1)
# print(masked_img)
# masked_img = cv2.inRange(masked_img, 150, 255)
# cv2.imwrite("masked.jpg", masked_img)

img = cv2.imread("original_imgs/segment.jpg", 0)
opt_threshold = optimal_thresholding(img)

img = threshold_img(img, opt_threshold)

img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

cv2.rectangle(img, (381, 37), (429, 258), (0,0,255), 2)
cv2.rectangle(img, (331, 19), (370, 295), (0,0,255), 2)
cv2.rectangle(img, (245, 72), (307, 211), (0,0,255), 2)
cv2.rectangle(img, (158, 122), (207, 170), (0,0,255), 2)

cv2.imwrite("output_imgs/segment_th.jpg", img)
#plot_histogram(img)
