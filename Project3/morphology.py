import cv2
import numpy as np

# Dilation
def dilate_img(img, structure):

    img_dilated = np.zeros(img.shape)
    height_img, width_img = img.shape
    height_struct, width_struct = len(structure), len(structure[0])
    struct_radius = height_struct//2

    for i in range(struct_radius, height_img-struct_radius):
        for j in range(struct_radius, width_img-struct_radius):

            slice_image = img[i-struct_radius:(i-struct_radius)+height_struct,
                                j-struct_radius:(j-struct_radius)+width_struct]

            height_slice, width_slice = slice_image.shape
            count = 0

            for m in range(0, height_slice):
                for n in range(0, width_slice):

                    if slice_image[m][n] == 1.0 and slice_image[m][n] == structure[m][n]:
                        count += 1
                        break;

            if count>0:
                img_dilated[i][j] = 1.0

    return img_dilated


# Erosion
def erode_image(img, structure):

    img_erosion = np.zeros(img.shape)
    height_img, width_img = img.shape
    height_struct, width_struct = len(structure), len(structure[0])
    struct_radius = height_struct//2

    for i in range(struct_radius, height_img-struct_radius):
        for j in range(struct_radius, width_img-struct_radius):

            slice_image = img[i-struct_radius:(i-struct_radius)+height_struct,
                              j-struct_radius:(j-struct_radius)+width_struct]

            height_slice, width_slice = slice_image.shape
            count_exp = sum(x.count(1) for x in structure)
            count_act = 0

            for m in range(0, height_slice):
                for n in range(0, width_slice):

                    if slice_image[m][n] == 1.0 and slice_image[m][n] == structure[m][n]:
                        count_act += 1

            if count_exp == count_act:
                img_erosion[i][j] = 1.0

    return img_erosion;


# Thresholding
def threshold_img(img):
    img = img/255
    return img


# Main
image1 = cv2.imread("original_imgs/noise.jpg", 0)
image1 = threshold_img(image1)

structure = [[1, 0, 0, 0, 1],
             [0, 1, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [1, 0, 0, 0, 1]]

img_dilated = dilate_img(image1, structure)
# cv2.imwrite("output_imgs/dilate.jpg", img_dilated*255)

img_erosion = erode_image(image1, structure)
# cv2.imwrite("output_imgs/erosion.jpg", img_erosion*255)

res_noise1 = erode_image(dilate_img(dilate_img(erode_image(image1, structure), structure), structure), structure)
cv2.imwrite("output_imgs/res_noise1.jpg", res_noise1*255)

res_noise2 = dilate_img(erode_image(erode_image(dilate_img(image1, structure), structure), structure), structure)
cv2.imwrite("output_imgs/res_noise2.jpg", res_noise2*255)

res_bound1 = dilate_img(res_noise1, structure) - erode_image(res_noise1, structure)
cv2.imwrite("output_imgs/res_bound1.jpg", res_bound1*255)

res_bound2 = dilate_img(res_noise2, structure) - erode_image(res_noise2, structure)
cv2.imwrite("output_imgs/res_bound2.jpg", res_bound2*255)
