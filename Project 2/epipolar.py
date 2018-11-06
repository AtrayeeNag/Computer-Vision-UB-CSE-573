import cv2
import sys
import colorsys
import numpy as np

UBIT = 'atrayeen'
np.random.seed(sum([ord(c) for c in UBIT]))

# Generates random colors to plot epipolar lines
def get_colors(num_colors):
    color = np.random.randint(0,255, size=(10, 3)).tolist()
    return color

# Extract keypoints of images using SIFT
def extract_keypoints(img, file_name):
    sift = cv2.xfeatures2d.SIFT_create()
    dummy = np.zeros((1,1))
    kp = sift.detect(img,None)
    img = cv2.drawKeypoints(img, kp, dummy)
    cv2.imwrite(file_name,img)

# Draw lines for matching keypoints of two images
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

# Calculatees the homography matrix for two images
def find_homography_matrix(img1, img2, kp1, kp2, match_without_list):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in match_without_list ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in match_without_list ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    return M, np.int32(src_pts), np.int32(dst_pts)

# Calculates the fundamental matrix for two matching image points
def find_fundamental_matrix(src_pts, dst_pts):
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
    src_pts_in = src_pts[mask.ravel()==1]
    dst_pts_in = dst_pts[mask.ravel()==1]
    return F, src_pts_in, dst_pts_in

# Utility to draw the epipolar lines with same color for matching pairs in two images
def drawlines(img1,img2,lines,pts1,pts2, color_list):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2,color in zip(lines,pts1,pts2,color_list):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1),tuple(color) ,1)
        img1 = cv2.circle(img1,tuple(pt1[0]),5,tuple(color),-1)
    return img1

# Generates the epipolar lines from left to right and from right to left.
def generate_epipolar_lines(img3, img4, color_list, F, src_pts_in, dst_pts_in):
    random_index = np.random.randint(low=0, high=src_pts_in.shape[0], size=10)
    src_pts_in = src_pts_in[random_index]
    dst_pts_in = dst_pts_in[random_index]
    lines1 = cv2.computeCorrespondEpilines(dst_pts_in.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5 = drawlines(img3, img4, lines1, src_pts_in, dst_pts_in, color_list)
    cv2.imwrite("task2_epi_left.jpg", img5)
    lines2 = cv2.computeCorrespondEpilines(src_pts_in.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img6 = drawlines(img4,img3,lines2,dst_pts_in,src_pts_in, color_list)
    cv2.imwrite("task2_epi_right.jpg", img6)

# generates the dispartiy image to show image depths.
def generate_disparity(img3, img4):
    window_size = 3
    min_disp = 16
    num_disp = 64-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
		numDisparities = num_disp,
		blockSize = 9,
		P1 = 8*3*window_size**2,
		P2 = 32*3*window_size**2,
		disp12MaxDiff = 1,
		uniquenessRatio = 10,
		speckleWindowSize = 100,
		speckleRange = 32
	)
    disparity = stereo.compute(img3, img4).astype(np.float32) / 16.0
    disparity = (disparity-min_disp)/num_disp
    disparity = disparity *250
    cv2.imwrite("task2_disparity.jpg", disparity)

img3 = cv2.imread("tsucuba_left.png", 0)
img4 = cv2.imread("tsucuba_right.png", 0)
color_list = get_colors(10)
extract_keypoints(img3, "task2_sift1.jpg")
extract_keypoints(img4, "task2_sift2.jpg")
kp1, kp2, match, match_without_list, src_pts, dst_pts = draw_match_image(img3, img4, "task2_matches_knn.jpg")
H, src_pts, dst_pts = find_homography_matrix(img3, img4, kp1, kp2, match_without_list)
F, src_pts_in, dst_pts_in = find_fundamental_matrix(src_pts, dst_pts)
print(F)
generate_epipolar_lines(img3, img4, color_list, F, src_pts_in, dst_pts_in)
generate_disparity(img3, img4)
