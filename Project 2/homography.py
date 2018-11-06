import cv2
import sys
import colorsys
import numpy as np
UBIT = 'atrayeen'
np.random.seed(sum([ord(c) for c in UBIT]))

# extracts the SIFT keypoints of an image
def extract_keypoints(img, file_name):
    sift = cv2.xfeatures2d.SIFT_create()
    dummy = np.zeros((1,1))
    kp = sift.detect(img,None)
    img = cv2.drawKeypoints(img, kp, dummy)
    cv2.imwrite(file_name,img)

# matches keypoints of 2 different images
def draw_match_image(img1, img2, file_name):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    dummy = np.zeros((1,1))
    match = []
    match_without_list =[]
    src_pts = []
    dst_pts = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            match.append([m])
            match_without_list.append(m)
            src_pts.append(kp1[m.queryIdx].pt)
            dst_pts.append(kp2[m.trainIdx].pt)
    img3 = np.zeros(img1.shape)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,match_without_list,img3,flags=2)
    cv2.imwrite(file_name,img3)
    return kp1, kp2, match, match_without_list, src_pts, dst_pts

# Calculatees the homography matrix and will draw 10 inlear matches between 2 images.
def find_homography_matrix_draw_match(img1, img2, file_name, kp1, kp2, match, match_without_list):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in match_without_list ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in match_without_list ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel()
    match_without_list_np = np.asarray(match_without_list)

    matchesMask = matchesMask[mask.ravel() == 1]
    match_without_list_np = match_without_list_np[mask.ravel() == 1]

    randIndx = np.random.randint(low=0, high=match_without_list_np.shape[0], size=10)

    matchesMask = matchesMask[randIndx]
    match_without_list_np = match_without_list_np[randIndx]

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask.tolist(), # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,match_without_list_np.tolist(),None,**draw_params)
    cv2.imwrite(file_name,img3)
    return M, np.int32(src_pts), np.int32(dst_pts)

def calculate_size(img1, img2, h):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    top_left = np.dot(h,np.asarray([0,0,1]))
    top_right = np.dot(h,np.asarray([w2,0,1]))
    bottom_left = np.dot(h,np.asarray([0,h2,1]))
    bottom_right = np.dot(h,np.asarray([w2,h2,1]))

    top_left = top_left/top_left[2]
    top_right = top_right/top_right[2]
    bottom_left = bottom_left/bottom_left[2]
    bottom_right = bottom_right/bottom_right[2]

    pano_left = int(min(top_left[0], bottom_left[0], 0))
    pano_right = int(max(top_right[0], bottom_right[0], w1))
    W = pano_right - pano_left

    pano_top = int(min(top_left[1], top_right[1], 0))
    pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
    H = pano_bottom - pano_top

    size = (W, H)

    X = int(min(top_left[0], bottom_left[0], 0))
    Y = int(min(top_left[1], top_right[1], 0))
    offset = (-X, -Y)
    return size, offset

def merge_images(img1, img2, h, size, offset, keypoints):
    (h1, w1) = img1.shape[:2]
    (h2, w2) = img2.shape[:2]
    panorama = np.zeros((size[1], size[0]), np.uint8)
    (ox, oy) = offset
    translation = np.matrix([
        [1.0, 0.0, ox],
        [0, 1.0, oy],
        [0.0, 0.0, 1.0]
        ])
    h = translation * h
    cv2.warpPerspective(img2, h, size, panorama)
    panorama[oy:h1+oy, ox:ox+w1] = img1
    return panorama

# Warps one image to another image using the matching points
def warp_images(img1, img2, file_name, H, src_pts, dst_pts):
    size, offset = calculate_size(img1, img2, H)
    img3 = merge_images(img2, img1, H, size, offset, (src_pts, dst_pts))
    cv2.imwrite(file_name, img3)

# Creates the fundamental matrix for two image matching points.
def find_fundamental_matrix(src_pts, dst_pts):
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
    src_pts_in = src_pts[mask.ravel()==1]
    dst_pts_in = dst_pts[mask.ravel()==1]
    return F, src_pts_in, dst_pts_in

img1 = cv2.imread("mountain1.jpg", 0)
img2 = cv2.imread("mountain2.jpg", 0)
extract_keypoints(img1, "task1_sift1.jpg")
extract_keypoints(img2, "task1_sift2.jpg")
kp1, kp2, match, match_without_list, src_pts, dst_pts = draw_match_image(img1, img2, "task1_matches_knn.jpg")
H, src_pts, dst_pts = find_homography_matrix_draw_match(img1, img2, "task1_matches.jpg", kp1, kp2, match, match_without_list)
print(H)
warp_images(img1, img2, "task1_pano.jpg", H, src_pts, dst_pts)
