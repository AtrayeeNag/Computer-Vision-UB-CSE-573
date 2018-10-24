import cv2
import math
from math import exp
from math import sqrt
import numpy as np

def gaussian(x, mu, sigma):
  return exp( -(((x-mu)/(sigma))**2)/2.0 )


def apply_sobel_filter(sobel, img, edge, kernel_radius):
    height, width = img.shape
    for x in range(kernel_radius, height-kernel_radius):
        for y in range(kernel_radius, width-kernel_radius):
            loop_end = (kernel_radius*2)+1
            sum = 0
            for i in range(0,loop_end):
                for j in range(0,loop_end):
                    sum += sobel[i][j] * img[x-kernel_radius+i][y-kernel_radius+j]

            edge[x][y] = sum
    return edge


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
    blur_image = [[0 for col in range(width)] for row in range(height)]
    blur_image = np.asarray(blur_image)
    blur_image = apply_sobel_filter(get_gaussian_kernel(sigma), img, blur_image, 3)
    return blur_image


def blurr_individual_images(img, octave_sigma, octave):
    for i in range(len(octave_sigma)):
        cv2.imwrite("blurr_"+octave+"_sigma"+str(i+1)+ ".jpg", apply_gaussian_blur(img, octave_sigma[i]))


def get_scaled_image(img, factor):
    x = 2
    height, width = img.shape
    octave_img = [[0 for col in range(int(width/2))] for row in range(int(height/2))]
    octave_x = 0
    for x in range(0, height):
        octave_y = 0
        if x%2 == 0:
            continue
        for y in range(0, width):
            if y%2 == 0:
                continue
            octave_img[octave_x][octave_y] = img[x][y]
            octave_y += 1
        octave_x += 1
    return np.asarray(octave_img)


def get_gauss_blurred_octaves(img, octave):
    octave1_sigma = np.array([1/sqrt(2), 1, sqrt(2), 2, 2*sqrt(2)])
    octave2_sigma = np.array([sqrt(2), 2, 2*sqrt(2), 4, 4*sqrt(2)])
    octave3_sigma = np.array([2*sqrt(2), 4, 4*sqrt(2), 8, 8*sqrt(2)])
    octave4_sigma = np.array([4*sqrt(2), 8, 8*sqrt(2), 16, 16*sqrt(2)])
    if octave == "octave1":
        blurr_individual_images(img, octave1_sigma, octave)
    if octave == "octave2":
        blurr_individual_images(img, octave2_sigma, octave)
    if octave == "octave3":
        blurr_individual_images(img, octave3_sigma, octave)
    if octave == "octave4":
        blurr_individual_images(img, octave4_sigma, octave)


def get_difference_of_gaussian(list_dog):
    for i in range(0, 4):
        for j in range(0, 4):
            img1 = cv2.imread("blurr_octave"+str(i+1)+"_sigma"+str(j+1)+".jpg", 0)
            img2 = cv2.imread("blurr_octave"+str(i+1)+"_sigma"+str(j+2)+".jpg", 0)
            height, width = img1.shape
            img3 = [[0 for col in range(width)] for row in range(height)]
            for x in range(0,height):
                for y in range(0, width):
                    img3[x][y] = int(img2[x][y]) - int(img1[x][y])
            img3 = np.asarray(img3)
            cv2.imwrite("octave_"+str(i+1)+"_dog"+str(j+1)+ ".jpg", img3)
            norm_img = cv2.normalize(img3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite("octave_"+str(i+1)+"_normalized_dog"+str(j+1)+ ".jpg", norm_img)
            list_dog.append(img3)
    return list_dog


def get_maxima_minima(dog_top, dog_middle, dog_bottom, scale, output_image):
    height, width = dog_middle.shape
    is_maxima = True
    is_minima = True
    for h in range(1, height-1):
        for w in range(1, width-1):
            if dog_middle[h][w]<2:
                continue
            is_maxima = True
            is_minima = True
            for x in range(h-1,h+2):
                for y in range(w-1,w+2):
                    if (dog_middle[h][w] < dog_middle[x][y]) or (dog_middle[h][w] < dog_top[x][y]) or (dog_middle[h][w] < dog_bottom[x][y]):
                        is_maxima = False
                        break

            for x in range(h-1,h+2):
                for y in range(w-1,w+2):
                    if (dog_middle[h][w] > dog_middle[x][y]) or (dog_middle[h][w] > dog_top[x][y]) or (dog_middle[h][w] > dog_bottom[x][y]):
                        is_minima = False
                        break
            if is_maxima or is_minima:
                output_image[h*scale][w*scale] = 255


def generate_keyPoints(list_dog):
    output_image = cv2.imread("keypoint.jpg", 0)
    get_maxima_minima(list_dog[0], list_dog[1], list_dog[2], 1, output_image)
    get_maxima_minima(list_dog[1], list_dog[2], list_dog[3], 1, output_image)
    get_maxima_minima(list_dog[4], list_dog[5], list_dog[6], 2, output_image)
    get_maxima_minima(list_dog[5], list_dog[6], list_dog[7], 2, output_image)
    get_maxima_minima(list_dog[8], list_dog[9], list_dog[10], 4, output_image)
    get_maxima_minima(list_dog[9], list_dog[10], list_dog[11], 4, output_image)
    get_maxima_minima(list_dog[12], list_dog[13], list_dog[14], 8, output_image)
    get_maxima_minima(list_dog[13], list_dog[14], list_dog[15], 8, output_image)
    cv2.imwrite("keypoint_detected.jpg", output_image)


def find_keypoints():
    img = cv2.imread("keypoint.jpg", 0)
    get_gauss_blurred_octaves(img, "octave1")
    octave2_image = get_scaled_image(img, 2)
    get_gauss_blurred_octaves(octave2_image, "octave2")
    octave3_image = get_scaled_image(octave2_image, 2)
    get_gauss_blurred_octaves(octave3_image, "octave3")
    octave4_image = get_scaled_image(octave3_image, 2)
    get_gauss_blurred_octaves(octave4_image, "octave4")
    list_dog = []
    list_dog = get_difference_of_gaussian(list_dog)
    generate_keyPoints(list_dog)


find_keypoints()
