import cv2
import math
from math import exp
from math import sqrt
import numpy as np


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


def edge_detection():
    img = cv2.imread("edge.png", 0)
    show_image(img)

    # Detect edge along x-direction
    sobel_x = [[-1,0,1],
               [-2,0,2],
               [-1,0,1]]

    height, width = img.shape
    edge_x = [[0.0 for col in range(width)] for row in range(height)]
    apply_sobel_filter(sobel_x, img, edge_x, 1)
    edge_x = np.asarray(edge_x)
    normalize_edge(edge_x)
    show_image(edge_x)

    # Detect edge along y-direction
    sobel_y = [[-1,-2,-1],
               [0,0,0],
               [1,2,1]]
    edge_y = [[0.0 for col in range(width)] for row in range(height)]
    apply_sobel_filter(sobel_y, img, edge_y, 1)
    edge_y = np.asarray(edge_y)
    normalize_edge(edge_y)
    show_image(edge_y)


# Normalize image by dividing absolute value of pixels with the max value
def normalize_edge(edge):
    height, width = edge.shape
    max_val=0
    for i in range(0, height):
        for j in range(0, width):
            edge[i][j] = abs(edge[i][j])
            max_val = max(max_val,edge[i][j])
    for i in range(0, height):
        for j in range(0, width):
            edge[i][j] = edge[i][j]/max_val


def show_image(img):
    cv2.namedWindow('blur_dir', cv2.WINDOW_NORMAL)
    cv2.imshow('blur_dir', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


edge_detection()
