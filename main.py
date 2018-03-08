#!/usr/bin/env python

import numpy as np
import cv2


def bgr_threshhold(img, b, g, r):
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


if __name__ == "__main__":
    print("Hello World")

    orig_img = cv2.imread("stop_sign1.jpeg")
    cv2.imshow("Original Image", orig_img)

    # Copy of original image. Do this before operating on an image because
    # img = orig_img does not copy the values, it copies the reference
    img = np.array(orig_img)

    # Image dimensions
    height = np.size(img, 0)
    width = np.size(img, 1)

    # Convert color space - img = [B G R] -> img_gray = [I]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur image for better lines
    img = cv2.GaussianBlur(img, (7, 7), 0)

    # Threshold image
    img_color_thresholded = bgr_threshhold(img, 80, 80, 110)
    cv2.imshow("Color Thresholded Image", img_color_thresholded)

    # Edge detection
    try:
        img_edges = np.array(img_color_thresholded)
        img_edges = cv2.cvtColor(img_edges, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_edges, 100, 200)
        cv2.imshow("Image Edges", img_edges)
    except Exception as e:
        print "Could not determine edges"

    # Contour finding
    try:
        contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img_contours = np.array(cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR))
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
        cv2.imshow("Image Contours", img_contours)
    except Exception as e:
        print "Could not determine counters"

    # Find largest contour
    try:
        largest_contour = []
        largest_size = 0
        for i in range(len(contours)):
            tmp = len(contours[i])
            if tmp > largest_size:
                largest_size = tmp
                largest_contour = contours[i]
        img_largest_contour = np.uint8(np.zeros([height, width]))
        for i in largest_contour:
            img_largest_contour[i[0][0]][i[0][1]] = 255
        cv2.imshow("Largest Contour", img_largest_contour)
    except Exception as e:
        print "Could not determine largest counter"

    # Hough transform
    try:
        lines = cv2.HoughLines(img_largest_contour, 1, np.pi/90, 40)
        points = []
        img_lines = cv2.cvtColor(img_largest_contour, cv2.COLOR_GRAY2BGR)
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = abs(int(a*rho))
            y0 = abs(int(b*rho))
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            points.append([x0, y0])
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Lines", img_lines)
    except Exception as e:
        print "Could not determine lines"

    # Perspective transform (To do)

    # Harris corner detection (when needed)
    #img_gray = np.float32(img_gray)
    #dst = cv2.cornerHarris(img_gray, 5, 3,0.04)
    #dst = cv2.dilate(dst, None)
    #img[dst > 0.2*dst.max()]=[0, 0, 255]

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Unused stuff. May be useful to keep for now
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)
    #laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    #sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    #sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    #img_edges = np.uint8(abs(np.hypot(sobel_x, sobel_y)))
    #img_edges = cv2.medianBlur(img_edges, 3)

    print("Goodbye World")
