import cv2
import numpy as np

#Dialation
def dialate_img(img, structure):

    img_dialated = np.zeros(img.shape)
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
                img_dialated[i][j] = 1.0

    return img_dialated


#Erosion
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


#Thresholding
def threshold_img(img):
    img = img/255
    return img



image1 = cv2.imread("original_imgs/noise.jpg", 0)
image1 = threshold_img(image1)

structure = [[1, 0, 0, 0, 1],
             [0, 1, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [1, 0, 0, 0, 1]]

img_dialated = dialate_img(image1, structure)
cv2.imwrite("output_imgs/dialate.jpg", img_dialated*255)

img_erosion = erode_image(image1, structure)
cv2.imwrite("output_imgs/erosion.jpg", img_erosion*255)

img_opening = dialate_img(erode_image(image1, structure), structure)
cv2.imwrite("output_imgs/res_noise1.jpg", img_opening*255)

img_closing = erode_image(dialate_img(image1, structure), structure)
cv2.imwrite("output_imgs/res_noise2.jpg", img_closing*255)

img_bound_op = dialate_img(img_opening, structure) - erode_image(img_opening, structure)
cv2.imwrite("output_imgs/res_bound1.jpg", img_bound_op*255)

img_bound_cl = dialate_img(img_closing, structure) - erode_image(img_closing, structure)
cv2.imwrite("output_imgs/res_bound2.jpg", img_bound_cl*255)
