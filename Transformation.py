import numpy as np
import cv2


def translate(image, x, y):
    m = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w//2, h//2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    dim = None
    if width is None and height is None:
        return image
    if width != None:
        r = width/float(w)
        dim = (width, int(h*r))
    else:
        r = height/float(h)
        dim = (int(w*r), height)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
