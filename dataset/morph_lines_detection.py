"""
@file morph_lines_detection.py
https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
@brief Use morphology transformations for extracting horizontal and vertical lines sample code
"""
import numpy as np
import sys
import cv2 as cv


def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)


def detect_verical_horizontal_lines(src, scale_percent=0.5, grids=[30, 30]):
    width = int(src.shape[1] * scale_percent)
    height = int(src.shape[0] * scale_percent)
    dim = (width, height)
    src = cv.resize(src, dim, interpolation=cv.INTER_AREA)
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src

    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    # horizontal
    cols = horizontal.shape[1]
    horizontal_size = cols // grids[0]
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)

    # vertical
    rows = vertical.shape[0]
    verticalsize = rows // grids[1]
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)

    return np.bitwise_or(horizontal, vertical)


def remove_lines(img):
    lines = detect_verical_horizontal_lines(img, 1.0)
    idx = (lines == 255)
    img[idx] = 255
    return img

if __name__ == "__main__":
    import glob
    scale = 1.0

    for path in glob.glob("/home/andn/Downloads/TableBank/TableBank_data/Detection_data/Latex/images/*.jpg"):
        img = cv.imread(path)
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        src = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        show_wait_destroy('raw', src)

        show_wait_destroy('line', detect_verical_horizontal_lines(img, scale))

        show_wait_destroy('removed_lines', remove_lines(img))

