#!/usr/bin/env python

import numpy as np
import cv2
print(cv2.__version__)
import cv_functions as cvf

# known points of a stop sign for a 500x500 image from top-left clockwise
STOPSIGN_KNOWNPOINTS = [[150., 5.], [350., 5.], \
                       [495., 150.], [495., 350.], \
                       [350., 495.], [150., 495.], \
                       [5., 350.], [5., 150.]]

TRIANGLESIGN_KNOWNPOINTS = [[250., 0.], [500., 500.], [0., 500.]]

if __name__ == "__main__":
    print("Hello World")

    orig_img = cv2.imread("triangle_sign4.jpg")
    cv2.imshow("Original Image", orig_img)

    # Copy of original image
    img = np.array(orig_img)

    # Making the image larger seems to help find the largest contour
    # need to determine best scale/ratio
    if img.shape[0] < 500 and img.shape[1] < 500:
        img = cv2.resize(img, None, fx=2, fy=2)
    elif img.shape[0] > 1000 and img.shape[1] > 1000:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)

    # Threshold image
    img = cvf.bgrThreshhold_red(img)
    cv2.imshow("Color Thresholded Image", img)

    img2, contours = cvf.getContours(img)
    cv2.imshow("Image Contours", img2)

    img, contour = cvf.getLargestContour(img, contours)
    cv2.imshow("Largest Contour", img)

    img = cvf.crop(img, contour)
    cv2.imshow("Cropped", img)

    img = cv2.GaussianBlur(img, (29, 29), 0)
    cv2.imshow("Blur", img)

    img = cvf.getEdges(img, 100, 200)
    cv2.imshow("Image Edges", img)

    img, lines = cvf.getHough(img, 60, 1, 100) # lowered theta threshold and overall threshold
    cv2.imshow("Lines", img)

    img, intersections = cvf.getIntersections(img, lines)
    cv2.imshow("Intersections", img)

    # Triangle sign
    intersections = cvf.triangleSign_sortIntersections(img, intersections)

    if orig_img.shape[0] < 500 and orig_img.shape[1] < 500:
        orig_img = cv2.resize(orig_img, None, fx=2, fy=2)
    elif orig_img.shape[0] > 1000 and orig_img.shape[1] > 1000:
        orig_img = cv2.resize(orig_img, None, fx=0.5, fy=0.5)
    orig_img = cvf.crop(orig_img, contour)
    img = cvf.triangleSign_affine(orig_img, intersections, TRIANGLESIGN_KNOWNPOINTS)
    cv2.imshow("Transformed image", img)

    # Stop sign
    # intersections = cvf.stopSign_sortIntersections(img, intersections)

    # if orig_img.shape[0] < 500 and orig_img.shape[1] < 500:
    #     orig_img = cv2.resize(orig_img, None, fx=2, fy=2)
    # elif orig_img.shape[0] > 1000 and orig_img.shape[1] > 1000:
    #     orig_img = cv2.resize(orig_img, None, fx=0.5, fy=0.5)
    # orig_img = cvf.crop(orig_img, contour)
    # img = cvf.stopSign_perspective(orig_img, intersections, STOPSIGN_KNOWNPOINTS)
    # cv2.imshow("Transformed image", img)

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

    # Blur image for better lines
    # img = cv2.GaussianBlur(img, (7, 7), 0)

    # img, corners, usable_corners = cvf.getCorners(img, 5, 5, 0.15, 0.05)
    # cv2.imshow("Corners", img)

    # does not work
    #img = cvf.clusterCorners(img, usable_corners, 2)
    #cv2.imshow("Clusters", img)
    # Perspective transform (To do)

    print("Goodbye World")
