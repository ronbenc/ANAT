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

# 1.a - Find High Correlation Location:
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
    corr_obj = np.asanyarray(corr_obj, np.float32)
    img = np.asanyarray(img, np.float32)

    max_obj = cv2.filter2D(src=corr_obj, ddepth = -1, kernel=corr_obj, borderType=cv2.BORDER_CONSTANT)
    min_val, max_val, match_coord, max_loc = cv2.minMaxLoc(max_obj)
    res = cv2.filter2D(src=img, ddepth = -1, kernel=corr_obj, borderType=cv2.BORDER_CONSTANT)
    # plt.imshow(np.abs(res-max_obj))
    # plt.show()

    min_val, max_val, match_coord, max_loc = cv2.minMaxLoc(np.abs(res-max_val))


    # ========================
    return match_coord

def main():
    # 1.b - Pre-Processing:
    vid_path = "HW2\given_data\Corsica.mp4" 
    frames = video_to_frames(vid_path=vid_path, start_second=250, end_second=260)
    transformed_frames_lst = []
    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transformed_frames_lst.append(transform(gray_frame))

    transformed_frames = np.asarray(transformed_frames_lst)
    
    # 1.c - Creating the panorama base
    early_frame_idx, ref_frame_idx, late_frame_idx = 20, 130, 200

    h,w = transformed_frames[0].shape
    panorama = np.zeros((h, int(w * 2.5)), np.uint8)
    ref_frame = transformed_frames[ref_frame_idx]
    panorama[:, int(((w * 2.5)//2)-w/2): int(((w * 2.5)//2)+w/2)] = ref_frame
    early_frame = transformed_frames[early_frame_idx]
    late_frame = transformed_frames[late_frame_idx]

    fig = plt.figure(figsize=(8, 5))
    plt.imshow(panorama, "gray")
    plt.title("updated panorama image")
    plt.axis('off')
    plt.show()

    fig, axs = plt.subplots(1,3)

    axs[0].imshow(early_frame, "gray")
    axs[0].set_title("Early Image")
    axs[0].set_axis_off()
    axs[1].imshow(ref_frame, "gray")
    axs[1].set_title("Reference Image")
    axs[1].set_axis_off()
    axs[2].imshow(late_frame, "gray")
    axs[2].set_title("Late Image")
    axs[2].set_axis_off()
    plt.show()
    # 1.d - Frames matching
    late_sub = late_frame[:,int(w*(4/5)):]
    early_sub = early_frame[:, :int(w/5)]

    corr_early_cor = match_corr(early_sub, ref_frame)
    corr_late_cor = match_corr(late_sub, ref_frame)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(early_sub, "gray")
    axs[0].set_title("Early frame" + str(corr_early_cor))
    axs[0].set_axis_off()
    axs[1].imshow(late_sub, "gray")
    axs[1].set_title("Late frame" + str(corr_late_cor))
    axs[1].set_axis_off()
    plt.show()

    # 1.e - It's panorama time!
    ref_start = int(0.75 * w ) 
    ref_end = int(0.75 * w ) + w
    
    late_start = ref_start + corr_late_cor[0] + int(w/10) - w
    late_end = ref_start + corr_late_cor[0] + int(w/10)

    early_start = ref_start + corr_late_cor[0] - int(w/10)
    early_end = ref_start + corr_late_cor[0] - int(w/10) + w

    panorama = np.asanyarray(panorama, np.float32)
    late_frame = np.asanyarray(late_frame, np.float32)
    # panorama[: , int(0.75 * w ) + corr_late_cor[0] + int(w/10) - w : int(0.75 * w ) + corr_late_cor[0] + int(w/10)] += 255
    
    panorama[: , late_start : late_end] += late_frame
    panorama[:, ref_start : late_end] //= 2
    panorama[: , early_start: early_end] += early_frame
    panorama[:, early_start : ref_end] //= 2
    # panorama[:, int(0.75 * w ): int(0.75 * w ) + corr_late_cor[1] + int(w/10)] = 0
    # panorama[: , int(1.25 * w ) + corr_early_cor[1] - int(w/10) : int(1.25 * w ) + corr_early_cor[1] - int(w/10) + w] += early_frame
    
    fig = plt.figure(figsize=(8, 5))
    plt.imshow(panorama, "gray")
    plt.title("Final panorama image")
    plt.axis('off')
    plt.show()
if __name__ == '__main__':
    main()