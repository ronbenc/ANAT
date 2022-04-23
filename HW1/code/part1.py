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
    frameCount = int(vidCap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = vidCap.get(cv2.CAP_PROP_FPS)

    all_frames_set = np.empty((frameCount, H, W, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, all_frames_set[fc] = vidCap.read()
        fc += 1

    vidCap.release()

    start_frame_idx = int(start_second * FPS)
    end_frame_idx = int(end_second * FPS)
    if start_frame_idx == end_frame_idx:
        frame_set = all_frames_set[start_frame_idx, :, :, :]
    else:
        frame_set = all_frames_set[start_frame_idx:end_frame_idx, :, :, :]

    # ========================

    return frame_set

if __name__ == '__main__':
    vid_path = "HW1\given_data\MilkyChance_StolenDance.mp4"
    start_second = 8
    end_second = 8
    frame_set = video_to_frames(vid_path, start_second, end_second)
    img = frame_set
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(grayscale_img, 'gray', vmin=0, vmax=255)
    plt.show()

    hist = plt.hist(grayscale_img.ravel(), bins=256, range=[0,255])
    plt.ylabel('count')
    plt.xlabel('pixel value')
    plt.title('Grayscale Histogram')
    plt.show()