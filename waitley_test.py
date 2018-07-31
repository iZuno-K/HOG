import cv2
from surf import load_compressed_image
if __name__ == '__main__':
    img = cv2.imread('data/10.jpg', 1)
    img = load_compressed_image(file='data/10.jpg', target_width=800)
    print(img.shape)
    cv2.imshow('window', img)
    while True:
        key = cv2.waitKey(30)
        if key == ord('a'):
            print('a!')
            break