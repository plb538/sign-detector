#!/usr/bin/env python

import numpy as np
import cv2

def bgrThreshhold_yellow(img):
    count = 0
    for i in img:
        for j in i:
            if int(j[2])-int(j[0]) > 90 and int(j[1])-int(j[0]) > 90:
                j[0] = 255
                j[1] = 255
                j[2] = 255
                count += 1
            else:
                j[0] = 0
                j[1] = 0
                j[2] = 0
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), count

def bgrThreshhold_red(img):
    count = 0
    for i in img:
        for j in i:
            if int(j[2])-int(j[0]) > 100 and int(j[2])-int(j[1]) > 100:
                j[0] = 255
                j[1] = 255
                j[2] = 255
                count += 1
            else:
                j[0] = 0
                j[1] = 0
                j[2] = 0
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), count


def colorThreshold(img, lh, ls, lv, uh, us, uv):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define range of color in HSV
        lower = np.array([lh, ls, lv])
        upper = np.array([uh, us, uv])

        # Threshold the HSV image
        mask = cv2.inRange(img, lower, upper)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask)
        img = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        count = np.sum(mask == 255)
        return img, mask, count
    except Exception as e:
        print "Could not color threshold image"
        return None, None


# Edge detection
def getEdges(img, t1, t2):
    try:
        # edges need intensity -> convert to gray
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Canny(img, t1, t2)
        return img
    except Exception as e:
        print "Could not determine edges"

# opening
def open(img, width, height):
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, height))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kern)
    return img

# closing
def close(img, width, height):
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, height))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kern)
    return img

# Contour finding
def getContours(img):
    try:
        #tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # contours need color -> convert to BGR
        img = np.array(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        return img, contours
    except Exception as e:
        print e
        print "Could not determine contours"


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
        # for i in largest_contour:
        #     img_largest_contour[i[0][1]][i[0][0]] = [0, 225, 0]
        img_largest_contour = cv2.drawContours(img, [largest_contour], -1, 255, -1)
        return img_largest_contour, largest_contour
    except Exception as e:
        print "Could not determine largest contour"

# crop to contour
def crop(img, contour):
    try:
        rect = cv2.boundingRect(contour)
        img = img[rect[1]-3:rect[1]+rect[3]+6, rect[0]-3:rect[0]+rect[2]+6]
        return cv2.resize(img, (500, 500))
    except Exception as e:
        print "Could not crop image"

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

# get intersection points
def getIntersections(img, lines):
    try:
        lin = []
        for i in range(0, len(lines)-1):
            rho1, theta1 = lines[i][0]
            for j in range(i+1, len(lines)):
                rho2, theta2 = lines[j][0]
                if (abs(rho1-rho2) < 50 and abs(theta1-theta2) < np.pi/18) or \
                   (abs(rho1+rho2) < 50 and abs(theta1-theta2) > np.pi-np.pi/18):
                    break
            else:
                lin.append(lines[i])
        lin.append(lines[-1])

        intersections = []
        for i in range(0, len(lin)-1):
            rho1, theta1 = lin[i][0]
            x1 = np.cos(theta1)
            y1 = np.sin(theta1)
            for j in range(i+1, len(lin)):
                rho2, theta2 = lin[j][0]
                x2 = np.cos(theta2)
                y2 = np.sin(theta2)

                p = np.cross([x1,y1,-rho1], [x2,y2,-rho2])

                if abs(p[2]) > 0.0001:
                    p = [p[0]/p[2], p[1]/p[2], 1]

                    intx = p[0]
                    inty = p[1]

                    if intx > -img.shape[1]/2 and intx < img.shape[1]*1.5 and \
                       inty > -img.shape[0]/2 and inty < img.shape[0]*1.5:
                        intersections.append([intx, inty])
                        cv2.circle(img, (int(intx), int(inty)), 3, [255,0,0], 3)
        return img, intersections
    except Exception as e:
        print "Could not determine intersections"

# sort yellow sign corners from top clockwise
def yellowSign_sortIntersections(img, intersections):
    sort = []

    xsort = sorted(intersections, lambda a,b: cmp(a[0], b[0]))
    ysort = sorted(intersections, lambda a,b: cmp(a[1], b[1]))

    sort.append(ysort[0])
    sort.append(xsort[-1])
    sort.append(ysort[-1])
    sort.append(xsort[0])

    imgc = img
    for i in range(0, len(sort)):
        cv2.putText(img, str(i), (sort[i][0], sort[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,255,0])
    cv2.imshow("labelled corners", imgc)

    return sort

# sort triangle sign corners from top clockwise for ^ and bottom anticlockwise for v
def triangleSign_sortIntersections(img, intersections):
    sort = []

    xsort = sorted(intersections, lambda a,b: cmp(a[0], b[0]))
    ysort = sorted(intersections, lambda a,b: cmp(a[1], b[1]))

    if ysort[1][1] > img.shape[0]/2:
        sort.append(ysort[0])
        sort.append(xsort[-1])
        sort.append(xsort[0])
    else:
        sort.append(ysort[-1])
        sort.append(xsort[-1])
        sort.append(xsort[0])

    imgc = img
    for i in range(0, len(sort)):
        cv2.putText(img, str(i), (sort[i][0], sort[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,255,0])
    #cv2.imshow("labelled corners", imgc)

    return sort

# sort stop sign corners from top left clockwise
def stopSign_sortIntersections(img, intersections):
    sort = []
    ints = []
    # get rid of unwanted intersections
    for i in intersections:
        # get rid of points near corners (within 75px x and y)
        if ((i[0] > 75 and i[0] < 425) or (i[1] > 75 and i[1] < 425)) \
            and i[0] > 0 and i[0] < img.shape[1] and i[1] > 0 and i[1] < img.shape[0]:
            for j in ints:
                # get rid of points that are within 10px of another
                if abs(i[0]-j[0]) < 10 and abs(i[1]-j[1]) < 10:
                    break
            else:
                ints.append(i)

    xsort = sorted(ints, lambda a,b: cmp(a[0], b[0]))
    ysort = sorted(ints, lambda a,b: cmp(a[1], b[1]))
    # top two points
    cand0 = ysort[0]
    cand1 = ysort[1]
    if cand0[0] > cand1[0]:
        cand0, cand1 = cand1, cand0
    sort.append(cand0)
    sort.append(cand1)

    # right two points
    cand0 = xsort[-1]
    cand1 = xsort[-2]
    if cand0[1] > cand1[1]:
        cand0, cand1 = cand1, cand0
    sort.append(cand0)
    sort.append(cand1)

    # bottom two points
    cand0 = ysort[-1]
    cand1 = ysort[-2]
    if cand0[0] < cand1[0]:
        cand0, cand1 = cand1, cand0
    sort.append(cand0)
    sort.append(cand1)

    # right two points
    cand0 = xsort[0]
    cand1 = xsort[1]
    if cand0[1] < cand1[1]:
        cand0, cand1 = cand1, cand0
    sort.append(cand0)
    sort.append(cand1)

    imgc = img
    for i in range(0, len(sort)):
        cv2.putText(img, str(i), (sort[i][0], sort[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,255,0])
    cv2.imshow("labelled corners", imgc)

    return sort

# perspective transform image using corresponding points
def yellowSign_perspective(img, img_ints, known_ints):
    tform0 = cv2.getPerspectiveTransform(np.array([img_ints[0], img_ints[1], img_ints[2], img_ints[3]], np.float32), \
                                         np.array([known_ints[0], known_ints[1], known_ints[2], known_ints[3]], np.float32))
    img = cv2.warpPerspective(img, tform0, (img.shape[0], img.shape[1]))
    return img

# affine transform image using corresponding points
def triangleSign_affine(img, img_ints, known_ints):
    if img_ints[1][1] < img.shape[0]/2:
        for i in known_ints:
            i[1] = img.shape[0]-i[1]
    tform0 = cv2.getAffineTransform(np.array([img_ints[0], img_ints[1], img_ints[2]], np.float32), \
                                    np.array([known_ints[0], known_ints[1], known_ints[2]], np.float32))
    img = cv2.warpAffine(img, tform0, (img.shape[0], img.shape[1]))
    return img

# perspective transform image using corresponding points
# tform requires 4 points, so do tform for two rectangles of stop sign points and take average
def stopSign_perspective(img, img_ints, known_ints):
    tform0 = cv2.getPerspectiveTransform(np.array([img_ints[0], img_ints[1], img_ints[4], img_ints[5]], np.float32), \
                                         np.array([known_ints[0], known_ints[1], known_ints[4], known_ints[5]], np.float32))
    tform1 = cv2.getPerspectiveTransform(np.array([img_ints[2], img_ints[3], img_ints[6], img_ints[7]], np.float32), \
                                         np.array([known_ints[2], known_ints[3], known_ints[6], known_ints[7]], np.float32))
    tform2 = (tform0+tform1)/2
    img = cv2.warpPerspective(img, tform2, (img.shape[0], img.shape[1]))
    return img

