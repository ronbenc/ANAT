import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img_path = "HW1\\my_data\\building.jpg"
    img = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    print(img.dtype)