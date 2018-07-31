import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

class MouseParam:
    def __init__(self, input_img_name):
        # mause input
        self.mouseEvent = {"x": None, "y": None, "event": None, "flags": None}
        # set callback
        cv2.setMouseCallback(input_img_name, self._CallBackFunc, None)

    # call back
    def _CallBackFunc(self, eventType, x, y, flags, userdata):
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

    # return mouse param
    def getData(self):
        return self.mouseEvent


def set_target(img):
    # specify object region and return the cropped image
    w_name = "window"
    cv2.imshow(w_name, img)

    mouseData = MouseParam(w_name)
    while 1:
        cv2.waitKey(20)
        if mouseData.getData()["event"] == cv2.EVENT_LBUTTONDOWN:
            start = [mouseData.getData()["y"], mouseData.getData()["x"]]
            print('start {}'.format(start))
        if mouseData.getData()["event"] == cv2.EVENT_LBUTTONUP:
            end = [mouseData.getData()["y"], mouseData.getData()["x"]]
            print('end {}'.format(end))
            break

    crop = img[start[0]:end[0], start[1]:end[1]]
    cv2.imshow(w_name, crop)
    for i in range(100):
        cv2.waitKey(20)

    return crop, np.array([start, end])


def load_compressed_image(file, target_width):
    img = cv2.imread(file, 0)
    # height width
    shape = img.shape
    aspect = shape[0] / shape[1]
    width = target_width
    reshape = (int(width / aspect), width)
    img = cv2.resize(img, reshape)

    return img


def background_color(img):
    # calc background color = average color to delete region
    return np.mean(img, axis=(0, 1))


if __name__ == '__main__':
    # read image in gray-scale
    img = load_compressed_image('data/5.jpg', 1000)
    print(img.shape)

    crop_image, coordinate = set_target(img)

    # load test image
    img = load_compressed_image('data/6.jpg', 1000)

    # # SURF detector
    hessian_threshold = 10
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    # detection and description
    kp1, des1 = surf.detectAndCompute(crop_image, None)  # query
    kp2, des2 = surf.detectAndCompute(img, None)  # test
    img2 = cv2.drawKeypoints(crop_image, kp1, None, (255, 0, 0), 4)
    plt.imshow(img2), plt.show()
    img2 = cv2.drawKeypoints(img, kp2, None, (255, 0, 0), 4)
    plt.imshow(img2), plt.show()

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(crop_image, kp1, img, kp2, good, None, flags=2)

    # img2 = cv2.drawKeypoints(img, kp1, None, (255, 0, 0), 4)
    plt.imshow(img3), plt.show()
