import numpy as numpy
import cv2
import matplotlib.pyplot as plt

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