import cv2
import sys
import colorsys
import numpy as np

UBIT = 'atrayeen'
np.random.seed(sum([ord(c) for c in UBIT]))

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

k_list = [20]
img1 = cv2.imread("baboon.jpg")

for k in range(len(k_list)):
    centroids = np.random.randint(0,255, size=(k_list[k], 3))

    centroid_list = []
    for r in range(5):
        k_labels = np.arange(k_list[k])
        cluster_list = []
        clusters = np.zeros([img1.shape[0], img1.shape[1]])
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                distances = dist(img1[i][j], centroids)
                cluster = np.argmin(distances)
                clusters[i][j] = cluster

        for m in range(len(k_labels)):
            image_pixel = []
            for i in range(img1.shape[0]):
                for j in range(img1.shape[1]):
                    if clusters[i][j] == m:
                        image_pixel.append(img1[i][j])
            cluster_list.append(np.asarray(image_pixel))

        centroids = []
        for c in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[c], axis=0))
        centroids = np.asarray(centroids)

    cluster_img = np.zeros(img1.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            cluster_img[i][j] = centroids[int(clusters[i][j])]

    cv2.imwrite("task3_baboon_"+str(k_list[k])+".jpg", cluster_img)
