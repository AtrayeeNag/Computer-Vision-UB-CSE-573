import cv2
import sys
import colorsys
import numpy as np
from matplotlib import pyplot as plt

UBIT = 'atrayeen'
np.random.seed(sum([ord(c) for c in UBIT]))

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

k = 3
colors = ['r', 'g', 'b']
rawData= np.array([[5.9, 3.2],
                    [4.6, 2.9],
                    [6.2, 2.8],
                    [4.7, 3.2],
                    [5.5, 4.2],
                    [5.0, 3.0],
                    [4.9, 3.1],
                    [6.7, 3.1],
                    [5.1, 3.8],
                    [6.0, 3.0]])
centroids = np.array([[6.2, 3.2],
                    [6.6, 3.7],
                    [6.5, 3.0]])

for m in range(2):
    clusters = np.zeros(len(rawData))
    color_array = []

    for i in range(len(rawData)):
        distances = dist(rawData[i], centroids)
        cluster = np.argmin(distances)
        clusters[i] = cluster

    for i in range(len(rawData)):
        for j in range(k):
            if clusters[i] == j:
                color_array.append(colors[j])

    print(np.int32(clusters))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(rawData[:,0], rawData[:,1], marker= "^", facecolors="None", edgecolors=color_array)
    plt.scatter(centroids[:,0], centroids[:,1], c= colors)
    for xy in zip(centroids[:,0], centroids[:,1]):
    	ax.annotate('(%.2f, %.2f)' % xy, xy=xy, textcoords='data')
    plt.savefig('task3_iter'+str(m+1)+'_a.jpg')
    plt.clf()

    for i in range(k):
        points = [rawData[j] for j in range(len(rawData)) if clusters[j] == i]
        centroids[i] = np.mean(points, axis=0)
    print(centroids)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(rawData[:,0], rawData[:,1], marker= "^", facecolors="None", edgecolors=color_array)
    plt.scatter(centroids[:,0], centroids[:,1], c=colors)
    for xy in zip(centroids[:,0], centroids[:,1]):
    	ax.annotate('(%.2f, %.2f)' % xy, xy=xy, textcoords='data')
    plt.savefig('task3_iter'+str(m+1)+'_b.jpg')
    plt.clf()
