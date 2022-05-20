import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # 2.a - Morphological operations
    img_path = "..\given_data\keyboard.jpg" 
    keyboard_img = cv2.imread(img_path)
    gray_keyboard_img = cv2.cvtColor(keyboard_img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_keyboard_img, "gray")
    plt.title("Keyboard image")
    plt.axis('off')
    plt.show()

    vert_line = np.ones((1,8))
    hor_line = vert_line.T
    erosion_vert = cv2.erode(gray_keyboard_img,vert_line,iterations = 1)
    erosion_hor = cv2.erode(gray_keyboard_img,hor_line,iterations = 1)

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(erosion_vert, "gray")
    axs[0].set_title("Erosion Vertical Image")
    axs[0].set_axis_off()

    axs[1].imshow(erosion_hor, "gray")
    axs[1].set_title("Erosion Horizontal Image")
    axs[1].set_axis_off()
    plt.show()

    erosion_sum = erosion_hor + erosion_vert
    threshold = 0.2*255
    ret,thresh1_img = cv2.threshold(erosion_sum, threshold, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh1_img, "gray")
    plt.title("Summation Image")
    plt.axis('off')
    plt.show()

    # 2.b - Median filtering
    inverse_img = cv2.bitwise_not(thresh1_img)
    median_img = cv2.medianBlur(inverse_img,9)
    plt.imshow(median_img, "gray")
    plt.title("Median Image")
    plt.axis('off')
    plt.show()

    # 2.c - Back to morphological operations
    third_kernel = np.ones((8, 8))
    eroded_img = cv2.erode(median_img,third_kernel,iterations = 1)
    plt.imshow(eroded_img, "gray")
    plt.title("Eroded Image")
    plt.axis('off')
    plt.show()
    

    # 2.d - Image sharpening and final thresholding
    eroded_img[eroded_img > 0] = 1
    inter_img = eroded_img * gray_keyboard_img
    sharpening_kerenl = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(src=inter_img, ddepth = -1, kernel=sharpening_kerenl, borderType=cv2.BORDER_CONSTANT)
    plt.imshow(sharpened_img, "gray")
    plt.title("Sharpened Image")
    plt.axis('off')
    plt.show()
    
    threshold = 250
    ret,thresh2_img = cv2.threshold(sharpened_img, threshold, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh2_img, "gray")
    plt.title("Thresholded Image ("+str(threshold)+")")
    plt.axis('off')
    plt.show()
    