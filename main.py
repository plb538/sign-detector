#!/usr/bin/env python

import numpy as np
import cv2
import cv_functions as cvf
import thinning as th

# known points of a stop sign for a 500x500 image from top-left clockwise
STOPSIGN_KNOWNPOINTS = [[150., 5.], [350., 5.], \
                       [495., 150.], [495., 350.], \
                       [350., 495.], [150., 495.], \
                       [5., 350.], [5., 150.]]

TRIANGLESIGN_KNOWNPOINTS = [[250., 0.], [500., 500.], [0., 500.]]

YELLOWSIGN_KNOWNPOINTS = [[250., 0.], [500., 250.], [250., 500.], [0., 250.]]


# Just moved your stuff here and copied as needed
def imageOperation(orig_img):

    orig_img = cv2.imread(orig_img)
    cv2.imshow("Original Image", orig_img)

    # Copy of original image
    img = np.array(orig_img)

    # Making the image larger seems to help find the largest contour
    # need to determine best scale/ratio
    if img.shape[0] < 500 and img.shape[1] < 500:
        img = cv2.resize(img, None, fx=2, fy=2)
    elif img.shape[0] > 1000 and img.shape[1] > 1000:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)

    cv2.imshow("Orignal image", img)
    cv2.waitKey(0)

    # Threshold image
    imgy, yellowCount = cvf.bgrThreshhold_yellow(img.copy())
    imgr, redCount = cvf.bgrThreshhold_red(img.copy())
    colour = ""
    if yellowCount > redCount:
        img = imgy
        colour = "yellow"
    else:
        img = imgr
        colour = "red"

    cv2.imshow("Color Thresholded Image", img)
    cv2.waitKey(0)

    img = cvf.close(img, 3, 3)
    cv2.imshow("Closing", img)
    cv2.waitKey(0)

    img2, contours = cvf.getContours(img)
    cv2.imshow("Image Contours", img2)
    cv2.waitKey(0)

    img, contour = cvf.getLargestContour(img2, contours)
    cv2.imshow("Largest Contour", img)
    cv2.waitKey(0)

    img = cvf.crop(img, contour)
    cv2.imshow("Cropped", img)
    cv2.waitKey(0)

    img = cv2.GaussianBlur(img, (29, 29), 0)
    cv2.imshow("Blur", img)
    cv2.waitKey(0)

    img = cvf.getEdges(img, 100, 200)
    cv2.imshow("Image Edges", img)
    cv2.waitKey(0)

    img, lines = cvf.getHough(img, 60, 1, 100)
    cv2.imshow("Lines", img)
    cv2.waitKey(0)

    img, intersections = cvf.getIntersections(img, lines)
    cv2.imshow("Intersections", img)
    cv2.waitKey(0)

    #print "Intersections: " + str(len(intersections))

    if orig_img.shape[0] < 500 and orig_img.shape[1] < 500:
        orig_img = cv2.resize(orig_img, None, fx=2, fy=2)
    elif orig_img.shape[0] > 1000 and orig_img.shape[1] > 1000:
        orig_img = cv2.resize(orig_img, None, fx=0.5, fy=0.5)
    orig_img = cvf.crop(orig_img, contour)

    if colour == "yellow":
        # Yellow Sign
        pass
        intersections = cvf.yellowSign_sortIntersections(img, intersections)

        img = cvf.yellowSign_perspective(orig_img, intersections, YELLOWSIGN_KNOWNPOINTS)
        cv2.imshow("Transformed image", img)

    elif colour == "red":
        # Triangle sign
        if len(intersections) == 3:
            intersections = cvf.triangleSign_sortIntersections(img, intersections)

            img = cvf.triangleSign_affine(orig_img, intersections, TRIANGLESIGN_KNOWNPOINTS)
            cv2.imshow("Transformed image", img)

        # Stop sign
        else:
            intersections = cvf.stopSign_sortIntersections(img, intersections)

            img = cvf.stopSign_perspective(orig_img, intersections, STOPSIGN_KNOWNPOINTS)
            cv2.imshow("Transformed image", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def videoOperation(v):
    cap = cv2.VideoCapture(v)
    fc = 0
    # Number of frames to average
    frame_buffer = 8
    # if we go through x number of frames but dont find 3 good signs, reset good sign count
    frames_to_check = 40
    # if we have a good sign, add it to the list
    good_sign = False
    good_sign_count = 0
    # if we find x number of good signs in a row, we clearly found a sign
    good_signs = []
    # set a previous color
    prev_colour = "red"
    while True:
        average_contour = np.zeros([500, 500], dtype=np.float32)
        for i in range(frame_buffer):
            try:
                fc += 1
                if fc % frames_to_check == 0:
                    good_sign_count = 0
                    good_signs = []
                ret, img = cap.read()
                if img is None:
                    print "FAILED"
                    exit(0)

                # very little difference in frames 1-5, so lets just call the last frame the original
                if i == frame_buffer - 1:
                    orig_img = img.copy()

                img = cv2.GaussianBlur(img, (9, 9), 0)

                # color thresholding using HSV space. Yellow may need to be tampered with
                imgr, maskr, countr = cvf.colorThreshold(img, 115, 110, 20, 179, 255, 255)
                imgy, masky, county = cvf.colorThreshold(img, 30, 50, 50, 40, 255, 255)

                if county > countr:
                    img = masky
                    colour = "yellow"
                else:
                    img = maskr
                    colour = "red"

                # Oh looks like we are starting with a yellow sign.
                # set previous sign to other color and restart
                if colour != prev_colour:
                    prev_colour = colour
                    good_sign = False
                    break

                img = cvf.close(img, 7, 7)

                img, contours = cvf.getContours(img)

                img, contour = cvf.getLargestContour(img, contours)

                img = cvf.crop(img, contour)

                img = cv2.GaussianBlur(img, (13, 13), 0)

                img = cvf.getEdges(img, 100, 200)

                average_contour = cv2.add(np.float32(img), average_contour)

            # If anything bad happens along the way, reset and begin again
            except Exception as e:
                good_sign = False
                break
        # Looks like nothing bad happened, lets continue
        else:
            try:
                img = np.uint8(average_contour/frame_buffer)

                img = cv2.GaussianBlur(img, (29, 29), 0)
                ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

                # Wasn't removing columns of pixels so I had to rotate
                img[0:5][:] = 0
                img[-1:-6][:] = 0
                M = cv2.getRotationMatrix2D((500/2, 500/2), 90, 1)
                img = cv2.warpAffine(img, M, (500, 500))
                img[0:5][:] = 0
                img[-1:-6][:] = 0
                M = cv2.getRotationMatrix2D((500/2, 500/2), -90, 1)
                img = cv2.warpAffine(img, M, (500, 500))

                # Found this online. super useful for thinning lines
                img = th.thinning(img)

                img, lines = cvf.getHough(img, 60, 1, 110)

                img, intersections = cvf.getIntersections(img, lines)

                orig_img = cvf.crop(orig_img, contour)
                if colour == "yellow":
                    # Yellow Sign
                    pass
                    final_intersections = cvf.yellowSign_sortIntersections(img, intersections)
                    img = cvf.yellowSign_perspective(orig_img, final_intersections, YELLOWSIGN_KNOWNPOINTS)

                elif colour == "red":
                    # Triangle sign
                    if len(intersections) == 3:
                        final_intersections = cvf.triangleSign_sortIntersections(img, intersections)
                        img = cvf.triangleSign_affine(orig_img, final_intersections, TRIANGLESIGN_KNOWNPOINTS)
                        if len(intersections) == 3 and len(final_intersections) == 3:
                            good_sign = True

                    # Stop sign
                    else:
                        final_intersections = cvf.stopSign_sortIntersections(img, intersections)
                        img = cvf.stopSign_perspective(orig_img, final_intersections, STOPSIGN_KNOWNPOINTS)
                        if len(intersections) == 16 and len(final_intersections) == 8:
                            good_sign = True
            # If anything bad happens along the way, reset and begin again
            except Exception as e:
                good_sign = False
        # So you've made it this far without any errors? Well you must have a good sign.
        # Add it to the list and incrememnt the count.
        # If we can get 3 of these, then we must have found a sign
        if good_sign is True:
            good_sign_count += 1
            good_signs.append(img)
        # 3 good signs found!!! Time to leave the loop
        if good_sign_count == 3:
            break
        # Still not done looking. Continue...
        else:
            continue

    cap.release()

    # Show our good signs
    for i in good_signs:
        cv2.imshow("Good image", i)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Hello World")

    # Stop signs
    ss_img1 = "stop_sign1.jpeg"
    ss_img2 = "stop_sign2.jpeg"
    ss_img3 = "stop_sign3.jpeg"
    ss_img4 = "stop_sign4.jpeg"

    # Red triangle signs
    trs_img1 = "triangle_sign1.jpeg"
    trs_img2 = "triangle_sign2.jpeg"
    trs_img3 = "triangle_sign3.jpeg"
    trs_img4 = "triangle_sign4.jpeg"

    # Yellow triangle signs
    tys_img1 = "yellow_sign1.jpeg"
    tys_img2 = "yellow_sign2.jpeg"
    tys_img3 = "yellow_sign3.jpeg"
    tys_img4 = "yellow_sign4.jpeg"

    # Stop sign videos
    ss_vid1 = "ss_vid1.mp4"
    ss_vid2 = "ss_vid2.mp4"
    ss_vid3 = "ss_vid3.mp4"
    ss_vid4 = "ss_vid4.mp4"
    ss_vid5 = "ss_vid5.mp4"
    ss_vid6 = "ss_vid6.mp4"

    
    imageOperation(ss_img3)
    imageOperation(ss_img4)
    imageOperation(trs_img4)
    imageOperation(trs_img2)
    imageOperation(tys_img4)
    imageOperation(tys_img3)

    cap = cv2.VideoCapture(ss_vid1)
    while True:
        ret, img = cap.read()
        cv2.imshow("Frame", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    videoOperation(ss_vid1)
    videoOperation(ss_vid2)
    videoOperation(ss_vid3)

    print("Goodbye World")
