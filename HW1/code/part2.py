import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import ifftshift


if __name__ == '__main__':
    img_path = "HW1\\my_data\\building.jpg"
    img = cv2.imread(img_path)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.dtype)

    dft_img = np.fft.fft2(grayscale_img)
    shifted_dft_img = np.fft.fftshift(dft_img)
    plt.imshow(np.log(1 + np.abs(shifted_dft_img)), "gray")
    plt.show()
