#!/usr/bin/env python

import numpy as np
import cv2 as cv


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

    img = cv.imread("stop_sign1.jpeg")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Threshold image
    img = bgr_threshhold(img, 80, 80, 110)

    img_gray = cv.Canny(img_gray, )

    # Harris corner detection
    img_gray = np.float32(img_gray)
    dst = cv.cornerHarris(img_gray, 5, 3,0.04)
    dst = cv.dilate(dst, None)
    img[dst > 0.01*dst.max()]=[0, 0, 255]

    # Perspective transform

    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("Goodbye World")
