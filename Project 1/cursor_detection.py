import cv2
import math
from math import exp
from math import sqrt
import numpy as np


def show_image(img):
    cv2.namedWindow('blur_dir', cv2.WINDOW_NORMAL)
    cv2.imshow('blur_dir', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def match_template(img, template, img_gaus_laplace, template_laplace):
    img2 = img_gaus_laplace.copy()
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    for m in methods:
        img = img2.copy()
        method = eval(m)
        # Apply template Matching on the blurred Laplacian image and the Laplacian template
        res = cv2.matchTemplate(img,template_laplace,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img,top_left, bottom_right, 255, 2)
        show_image(img)

def find_cursor():
    template_1 = cv2.imread('template.png',0)
    template_laplace_1 = cv2.Laplacian(template_1,cv2.CV_8U)
    # Check cursor for positive images for Set A
    for i in range(1,16):
        print("Positive images")
        img_1_pos = cv2.imread("pos_"+str(i)+".jpg",0)
        img_gaus_1_pos = cv2.GaussianBlur(img_1_pos, (3,3), 0)
        img_gaus_laplace_1_pos = cv2.Laplacian(img_gaus_1_pos,cv2.CV_8U)
        match_template(img_1_pos, template_1, img_gaus_laplace_1_pos, template_laplace_1);

    # Check cursor for negative images for Set B
    for i in range(1,10):
        print("Negative images")
        img_1_neg = cv2.imread("neg_"+str(i)+".jpg",0)
        img_gaus_1_neg = cv2.GaussianBlur(img_1_neg, (3,3), 0)
        img_gaus_laplace_1_neg = cv2.Laplacian(img_gaus_1_neg,cv2.CV_8U)
        match_template(img_1_neg, template_1, img_gaus_laplace_1_neg, template_laplace_1);

    #Check cursor for task 3 Set B
    for i in range(1, 4):
        template_2 = cv2.imread("t"+str(i)+".jpg",0)
        # Perform Laplacian on the template
        template_laplace_2 = cv2.Laplacian(template_2,cv2.CV_8U)
        for j in range(1,7):
            img_2 = cv2.imread("t"+str(i)+"_"+str(j)+".jpg",0)
            # Perform Gaussian and Laplacian on the image
            img_gaus_2 = cv2.GaussianBlur(img_2, (3,3), 0)
            img_gaus_laplace_2 = cv2.Laplacian(img_gaus_2,cv2.CV_8U)
            match_template(img_2, template_2, img_gaus_laplace_2, template_laplace_2);

    # Check cursor for negative images for Task 3 Set B
    for i in range(1, 4):
        template_2 = cv2.imread("t"+str(i)+".jpg",0)
        # Perform Laplacian on the template
        template_laplace_2 = cv2.Laplacian(template_2,cv2.CV_8U)
        for i in range(1,10):
            print("Negative images")
            img_1_neg = cv2.imread("neg_"+str(i)+".jpg",0)
            img_gaus_1_neg = cv2.GaussianBlur(img_1_neg, (3,3), 0)
            img_gaus_laplace_1_neg = cv2.Laplacian(img_gaus_1_neg,cv2.CV_8U)
            match_template(img_1_neg, template_1, img_gaus_laplace_1_neg, template_laplace_1);

find_cursor()
