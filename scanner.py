from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

if __name__ == '__main__':
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    image_file_name = "data/receipt.jpg"
    # image_file_name = "data/recepit2.jpg"
    image = cv2.imread(image_file_name)
    gray_img = image.copy()
    warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=30, method="gaussian")
    warped = (warped > T).astype("uint8") * 255
    mser = cv2.MSER_create()
    regions = mser.detectRegions(warped)

    # This is the countours
    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]

    #This is the bounding box of all the characters / mser's.
    boundingBoxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions[0]]
    for (x,y,w,h) in boundingBoxes:
        cv2.rectangle(gray_img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Method to show countours
    # cv2.polylines(gray_img, hulls, 1, (0, 255, 0))

    cv2.namedWindow('img', 0)
    cv2.imshow('img', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()