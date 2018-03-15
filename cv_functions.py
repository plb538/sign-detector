#!/usr/bin/env python

import numpy as np
import cv2

def bgrThreshhold(img, b, g, r):
    for i in img:
        for j in i:
            if j[0] < b and j[1] < g and j[2] > r:
                j[0] = 255
                j[1] = 255
                j[2] = 255
            else:
                j[0] = 0
                j[1] = 0
                j[2] = 0
    return img


# Edge detection
def getEdges(img, t1, t2):
    try:
        # edges need intensity -> convert to gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Canny(img, t1, t2)
        return img
    except Exception as e:
        print "Could not determine edges"


# Contour finding
def getContours(img):
    try:
        img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # contours need color -> convert to BGR
        img = np.array(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        return img, contours
    except Exception as e:
        print e
        print "Could not determine counters"


# Find largest contour
def getLargestContour(img, contours):
    try:
        largest_contour = []
        largest_size = 0
        for i in range(len(contours)):
            tmp = len(contours[i])
            if tmp > largest_size:
                largest_size = tmp
                largest_contour = contours[i]
        img_largest_contour = np.uint8(np.zeros([np.size(img, 0), np.size(img, 1)]))
        for i in largest_contour:
            img_largest_contour[i[0][0]][i[0][1]] = 255
        return img_largest_contour
    except Exception as e:
        print "Could not determine largest counter"


# Hough transform
def getHough(img, thresh, r_acc, t_acc):
    try:
        lines = cv2.HoughLines(img, r_acc, np.pi/t_acc, thresh)
        # lines need color -> copy largest contour and convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = int(a*rho)
            y0 = int(b*rho)
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return img, lines
    except Exception as e:
        print "Could not determine lines"


# Harris corner detection
def getCorners(img, bsize, ksize, k, t):
    try:
        tmp = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerHarris(tmp, bsize, ksize, k)
        corners = cv2.dilate(corners, None)
        usable_corners = corners > t*corners.max()
        # turn corners blue
        img[usable_corners] = [255, 0, 0]
        return img, corners, usable_corners
    except Exception as e:
        print "Could not determine corners"


# Cluster corner points
# does not work
def clusterCorners(img, usable_corners, num_clusters):
    try:
        tmp = np.uint8(np.zeros([np.size(img, 0), np.size(img, 1)]))
        tmp[usable_corners] = 255
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(np.float32(tmp), num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        print center[1,:]
        res = center[label.flatten()]
        res2 = res.reshape(img.shape)

        return res2
    except Exception as e:
        print "Could not cluster corner points"
