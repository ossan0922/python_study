import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def make_image(input_img):
    img_size = input_img.shape
    filter_one = np.ones((3,3))

    mat1 = cv2.getRotationMatrix2D(tuple(np.array(input_img.shape[:2]) / 2), 23, 1)
    mat2 = cv2.getRotationMatrix2D(tuple(np.array(input_img.shape[:2]) / 2), 144, 0.8)

    fake_method_array = np.array([
        lambda image: cv2.warpAffine(image, mat1, image.shape[:2]),
        lambda image: cv2.warpAffine(image, mat2, image.shape[:2]),
        lambda image: cv2.threshold(image, 100, 255, cv2.THRESH_TOZERO)[1],
        lambda image: cv2.GaussianBlur(image, (5, 5), 0),
        lambda image: cv2.resize(cv2.resize(image, (img_size[1] // 5, img_size[0] // 5)),(img_size[1], img_size[0])),
        lambda image: cv2.erode(image, filter_one),
        lambda image: cv2.flip(image, 1)
    ])

    images = []

    for method in fake_method_array:
        faked_img = method(input_img)
        images.append(faked_img)

    return images

target_img = cv2.imread("1.jpg")

fake_images = make_image(target_img)

if not os.path.exists("fake_images"):
    os.mkdir("fake_images")

for number, img in enumerate(fake_images):
    cv2.imwrite("fake_images/" + str(number) + ".jpg", img)
    
NUM_COLUMNS = 4

ROWS_COUNT = len(fake_images) % NUM_COLUMNS

COLUMNS_COUNT = NUM_COLUMNS

subfig = []

fig = plt.figure(figsize=(12,9))

for i in range(1, len(fake_images) + 1):
    subfig.append(fig.add_subplot(ROWS_COUNT, COLUMNS_COUNT, i))

    img_bgr = cv2.imread('fake_images/' + str(i-1) + '.jpg')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    subfig[i-1].imshow(img_rgb)

fig.subplots_adjust(wspace=0.3, hspace=0.3)

plt.show()