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
    cv2.imshow("Orignal Image", orig_img)

    # Get copy of image
    img = orig_img[:]

    # Image dimensions
    height = np.size(img, 0)
    width = np.size(img, 1)

    # Threshold image
    img = bgr_threshhold(img, 80, 80, 110)
    cv2.imshow("Color Thresholded Image", img)

    # Convert color space - img = [B G R] -> img_gray = [I]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # fooled around with Canny params - could not see difference in output images
    img_canny = cv2.Canny(img_gray, 100, 200)
    cv2.imshow("Image Edges", img_canny)

    # Contour collection here. add to hough below

    # Hough Transform
    lines = cv2.HoughLines(img_canny, 1, np.pi/180, 40)
    points = []
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        points.append([])

    # Display lines
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Lines", img)


    # Harris corner detection
    #img_gray = np.float32(img_gray)
    #dst = cv2.cornerHarris(img_gray, 5, 3,0.04)
    #dst = cv2.dilate(dst, None)
    #img[dst > 0.2*dst.max()]=[0, 0, 255]

    # Perspective transform

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Unused stuff. May be useful to keep for now
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)
    #laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    #sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    #sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    #img_edges = cv2.medianBlur(img_edges, 3)


    print("Goodbye World")
