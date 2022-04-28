import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # section a
    parrot_path = "HW1\given_data\parrot.png"
    parrot_img = cv2.imread(parrot_path, cv2.IMREAD_GRAYSCALE)
    selfie_path = "HW1\my_data\passport.jpg"
    selfie_img = cv2.resize(cv2.imread(selfie_path, cv2.IMREAD_GRAYSCALE), parrot_img.shape)
    print(parrot_img.dtype)
    print(selfie_img.dtype)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(parrot_img, "gray")
    axs[1].imshow(selfie_img, "gray")
    axs[0].set_title("Parrot image")
    axs[1].set_title("Selfie image")
    plt.show()

    # section b
    dft_parrot_img = np.fft.fft2(parrot_img)
    dft_selfie_img = np.fft.fft2(selfie_img)

    phase_parrot_img = np.angle(dft_parrot_img)
    phase_selfie_img = np.angle(dft_selfie_img)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(phase_parrot_img, "gray")
    axs[1].imshow(phase_selfie_img, "gray")
    axs[0].set_title("Parrot phase image")
    axs[1].set_title("Selfie phase image")
    plt.show()

    amp_parrot_img = np.abs(dft_parrot_img)
    amp_selfie_img = np.abs(dft_selfie_img)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.log(1 + np.abs(amp_parrot_img)), "gray")
    axs[1].imshow(np.log(1 + np.abs(amp_selfie_img)), "gray")
    axs[0].set_title("Parrot amplitude image")
    axs[1].set_title("Selfie amplitude image")
    plt.show()

    # part c