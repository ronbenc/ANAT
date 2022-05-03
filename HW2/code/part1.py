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

    
# def video_to_frames(vid_path: str, start_second, end_second):
#         """
#         Load a video and return its frames from the wanted time range.
#         :param vid_path: video file path.
#         :param start_second: time of first frame to be taken from the
#         video in seconds.
#         :param end_second: time of last frame to be taken from the
#         video in seconds.
#         :return:
#         frame_set: a 4D uint8 np array of size [num_of_frames x H x W x C]
#         containing the wanted video frames in BGR format.
#         """
#         vid_cap = cv2.VideoCapture(vid_path)
#         frameCount = end_second - start_second
#         frameWidth = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frameHeight = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         frame_set = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
#         i = 0
#         while True:
#                 ret, frame = vid_cap.read()
#                 if (i >= start_second) and (i < end_second):
#                         frame_set[i-start_second] = frame
#                 if (i > end_second):
#                         vid_cap.release()
#                         break
#                 i += 1
#         return frame_set

def transform(img: np.ndarray)->np.ndarray:
    img_rows = img.shape[0]
    transformed_img = img[int(img_rows*(1/3)):, 7:627]
    return transformed_img

def match_corr(corr_obj, img):
    """
    return the center coordinates of the location of 'corr_obj' in 'img'.
    :param corr_obj: 2D numpy array of size [H_obj x W_obj]
    containing an image of a component.
    :param img: 2D numpy array of size [H_img x W_img]
    where H_img >= H_obj and W_img>=W_obj,
    containing an image with the 'corr_obj' component in it.
    :return:
    match_coord: the two center coordinates in 'img'
    of the 'corr_obj' component.
    """
    # ====== YOUR CODE: ======
    max_obj = np.max(cv2.filter2D(src=corr_obj, ddepth = -1, kernel=corr_obj, borderType=cv2.BORDER_CONSTANT))
    res = cv2.filter2D(src=img, ddepth = -1, kernel=corr_obj, borderType=cv2.BORDER_CONSTANT)

    min_val, max_val, min_loc, match_coord = cv2.minMaxLoc(np.abs(res-max_obj))

    # ========================
    return match_coord

def main():
    vid_path = "HW2\given_data\Corsica.mp4"
    fps = 25    
    #frames = video_to_frames(vid_path=vid_path, start_second=250*fps, end_second=260*fps)
    frames = video_to_frames(vid_path=vid_path, start_second=250, end_second=260)
    # transformed_frames = np.zeros(frames.shape[cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).shape(),
    transformed_frames_lst = []
    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # transformed_frames[idx] = transform(gray_frame)
        transformed_frames_lst.append(transform(gray_frame))

    transformed_frames = np.asarray(transformed_frames_lst)
    plt.imshow(transformed_frames[0], cmap='gray')

    plt.show()

if __name__ == '__main__':
    main()