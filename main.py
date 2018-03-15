#!/usr/bin/env python

import numpy as np
import cv2
print(cv2.__version__)
import cv_functions as cvf

if __name__ == "__main__":
    print("Hello World")

    orig_img = cv2.imread("stop_sign1.jpeg")
    cv2.imshow("Original Image", orig_img)

    # Copy of original image
    img = np.array(orig_img)

    # Making the image larger seems to help find the largest contour
    # need to determine best scale/ratio
    img = cv2.resize(img, None, fx=2, fy=2)

    # Threshold image
    img = cvf.bgrThreshhold(img, 100, 100, 110)
    cv2.imshow("Color Thresholded Image", img)

    img = cvf.getEdges(img, 100, 200)
    cv2.imshow("Image Edges", img)

    img, contours = cvf.getContours(img)
    cv2.imshow("Image Contours", img)

    img = cvf.getLargestContour(img, contours)
    cv2.imshow("Largest Contour", img)

    img, lines = cvf.getHough(img, 50, 1, 180)
    cv2.imshow("Lines", img)

    # Blur image for better lines
    img = cv2.GaussianBlur(img, (7, 7), 0)

    img, corners, usable_corners = cvf.getCorners(img, 5, 5, 0.15, 0.05)
    cv2.imshow("Corners", img)

    # does not work
    #img = cvf.clusterCorners(img, usable_corners, 2)
    #cv2.imshow("Clusters", img)
    # Perspective transform (To do)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Unused stuff. May be useful to keep for now
    #kernel = cv2.cvf.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)
    #laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    #sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    #sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    #img_edges = np.uint8(abs(np.hypot(sobel_x, sobel_y)))
    #img_edges = cv2.medianBlur(img_edges, 3)

    print("Goodbye World")
