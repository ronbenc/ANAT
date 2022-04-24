from filecmp import cmp
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import ifftshift


def max_freq_filtering(fshift, precentege):
 """
 Reconstruct an image using only its maximal amplitude frequencies.
 :param fshift: The fft of an image, **after fftshift** -
 complex float ndarray of size [H x W].
 :param precentege: the wanted precentege of maximal frequencies.
 :return:
 fMaxFreq: The filtered frequency domain result -
 complex float ndarray of size [H x W].
 imgMaxFreq: The filtered image - real float ndarray of size [H x W].
 """
 # ====== YOUR CODE: ======

 # ========================
 return fMaxFreq, imgMaxFreq




if __name__ == '__main__':
    img_path = "HW1\\my_data\\building.jpg"
    img = cv2.imread(img_path)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.dtype)


    dft_img = np.fft.fft2(grayscale_img)
    shifted_dft_img = np.fft.fftshift(dft_img)
    plt.imshow(np.log(1 + np.abs(shifted_dft_img)), "gray")
    plt.title("2D-DFT of Tehcnion building")
    plt.show()


    center_row_idx = shifted_dft_img.shape[0]//2
    center_col_idx = shifted_dft_img.shape[1]//2
    number_of_passed_freq_k = int(0.02*shifted_dft_img.shape[0])
    number_of_passed__freq_l = int(0.02*shifted_dft_img.shape[1])

    l_filtered_dft = np.zeros_like(shifted_dft_img)
    l_filtered_dft[:, center_col_idx-number_of_passed__freq_l//2: center_col_idx+number_of_passed__freq_l//2] = shifted_dft_img[:,  center_col_idx-number_of_passed__freq_l//2: center_col_idx+number_of_passed__freq_l//2]

    k_filtered_dft = np.zeros_like(shifted_dft_img)
    k_filtered_dft[center_row_idx-number_of_passed__freq_l//2: center_row_idx+number_of_passed__freq_l//2, :] = shifted_dft_img[center_row_idx-number_of_passed__freq_l//2: center_row_idx+number_of_passed__freq_l//2, :]

    lk_filtered_dft = np.zeros_like(shifted_dft_img)
    lk_filtered_dft[:, center_col_idx-number_of_passed__freq_l//2: center_col_idx+number_of_passed__freq_l//2] = shifted_dft_img[:,  center_col_idx-number_of_passed__freq_l//2: center_col_idx+number_of_passed__freq_l//2]
    lk_filtered_dft[center_row_idx-number_of_passed__freq_l//2: center_row_idx+number_of_passed__freq_l//2, :] = shifted_dft_img[center_row_idx-number_of_passed__freq_l//2: center_row_idx+number_of_passed__freq_l//2, :]

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(np.log(1 + np.abs(l_filtered_dft)), "gray")
    axs[1].imshow(np.log(1 + np.abs(k_filtered_dft)), "gray")
    axs[2].imshow(np.log(1 + np.abs(lk_filtered_dft)), "gray")
    axs[0].set_title("(1)")
    axs[1].set_title("(2)")
    axs[2].set_title("(3)")
    plt.show()

    l_filtered_dft_not_centered = np.fft.ifftshift(l_filtered_dft)
    k_filtered_dft_not_centered = np.fft.ifftshift(k_filtered_dft)
    lk_filtered_dft_not_centered = np.fft.ifftshift(lk_filtered_dft)
    l_reconstructed = np.abs(np.fft.ifft2(l_filtered_dft_not_centered))
    k_reconstructed = np.abs(np.fft.ifft2(k_filtered_dft_not_centered))
    lk_reconstructed = np.abs(np.fft.ifft2(lk_filtered_dft_not_centered))

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(l_reconstructed, "gray")
    axs[1].imshow(k_reconstructed, "gray")
    axs[2].imshow(lk_reconstructed, "gray")
    axs[0].set_title("Reconstructed (1)")
    axs[1].set_title("Reconstructed (2)")
    axs[2].set_title("Reconstructed (3)")
    plt.show()
