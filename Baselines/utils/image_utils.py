import cv2


def resize(image, img_w, img_h, interpolation=cv2.INTER_AREA):
    dim = (img_w, img_h)
    # resize image
    resized = cv2.resize(image, dim, interpolation=interpolation)
    return resized