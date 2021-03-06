import cv2
import numpy as np
from matplotlib import pyplot as plt

def video_to_frames(vid_path: str, start_second, end_second):
    """
    Load a video and return its frames from the wanted time range.
    :param vid_path: video file path.
    :param start_second: time of first frame to be taken from the
    video in seconds.
    :param end_second: time of last frame to be taken from the
    video in seconds.
    :return:
    frame_set: a 4D uint8 np array of size [num_of_frames x H x W x C]
    containing the wanted video frames in BGR format.
    """
    # ====== YOUR CODE: ======
    vidCap = cv2.VideoCapture(vid_path)
    H = int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FPS = vidCap.get(cv2.CAP_PROP_FPS)

    start_frame_idx = int(start_second * FPS)
    end_frame_idx = int(end_second * FPS)
    
    frame_set = np.empty((end_frame_idx - start_frame_idx + 1, H, W, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc <= end_frame_idx and ret):
        ret, frame = vidCap.read()
        if fc >= start_frame_idx:
            frame_set[fc-start_frame_idx] = frame
        fc += 1

    vidCap.release()

    # ========================

    return frame_set

def gamma_correction(img, gamma):
    """
    Perform gamma correction on a grayscale image.
    :param img: An input grayscale image - ndarray of uint8 type.
    :param gamma: the gamma parameter for the correction.
    :return:
    gamma_img: An output grayscale image after gamma correction -
    uint8 ndarray of size [H x W x 1].
    """
    # ====== YOUR CODE: ======
    gamma_img = 255*(np.power(img/255, gamma))
    # ========================
    return gamma_img


def frames_mean(frame_set):
    """
    Calculate the mean frame out of a set of frames.
    :param frame_set: a 4D uint8 np array of size [num_of_frames x H x W x C].
    :return:
    mean_frame: the average image calculated from the set,
    a 3D uint8 np array of size [H x W x C].
    """
    # ====== YOUR CODE: ======
    mean_frame = np.array(np.mean(frame_set, axis=0), dtype=np.uint8)
    # ========================
    return mean_frame
def frames_median(frame_set):
    """
    Calculate the median frame out of a set of frames.
    :param frame_set: a 4D uint8 np array of size [num_of_frames x H x W x C].
    :return:
    median_frame: the median image calculated from the set,
    a 3D uint8 np array of size [H x W x C].
    """
    # ====== YOUR CODE: ======
    median_frame = np.array(np.median(frame_set, axis=0), dtype=np.uint8)
    # ========================
    return median_frame

def plot_grayscale_img(grayscale_img, title):
    plt.title(title)
    plt.imshow(grayscale_img, 'gray')
    plt.show()

def plot_grayscale_hist(grayscale_img, title):
    hist = plt.hist(grayscale_img.ravel(), bins=256, range=[0,255])
    plt.ylabel('count')
    plt.xlabel('pixel value')
    plt.title(title)
    plt.show()

def plot_color_img(color_img, title):
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()      

if __name__ == '__main__':
    vid_path = "HW1\given_data\MilkyChance_StolenDance.mp4"
    start_second = 8
    end_second = 8
    frame_set = video_to_frames(vid_path, start_second, end_second)
    img = frame_set[0]
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plot_grayscale_img(grayscale_img, title="Grayscale Image of 8th second frame")
    plot_grayscale_hist(grayscale_img, title="Grayscale Histogram")

    for gamma in [0.5, 1.5]:
        gamma_corrected_img = gamma_correction(grayscale_img, gamma)
        plot_grayscale_img(gamma_corrected_img, title="Gamma = " + str(gamma))
        plot_grayscale_hist(gamma_corrected_img, title="Gamma = " + str(gamma))


    start_second = 45
    end_second = 48
    frame_set = video_to_frames(vid_path, start_second, end_second)
    color_frame_idx = np.random.randint(75)
    img = frame_set[color_frame_idx]
    plot_color_img(img, title="Selected color image from frame numer " + str(45*25 + color_frame_idx))

    meam_img = frames_mean(frame_set)
    median_img = frames_median(frame_set)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(cv2.cvtColor(meam_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("image from the mean")
    axs[1].imshow(cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB))
    axs[1].set_title("image from the median")
    plt.show()