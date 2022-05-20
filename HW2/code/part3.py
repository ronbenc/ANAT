from attr import field
import cv2
import numpy as np
from matplotlib import pyplot as plt


'1.A'
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
        vid_cap = cv2.VideoCapture(vid_path)
        frameCount = end_second - start_second
        frameWidth = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_set = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        i = 0
        while True:
                ret, frame = vid_cap.read()
                if (i >= start_second) and (i < end_second):
                        frame_set[i-start_second] = frame
                if (i > end_second):
                        vid_cap.release()
                        break
                i += 1
        return frame_set
# def poisson_noisy_image(X, a):
#     """
#     Creates a Poisson noisy image.
#     :param X: The Original image. np array of size [H x W] and of type uint8.
#     :param a: number of photons scalar factor
#     :return:
#     Y: The noisy image. np array of size [H x W] and of type uint8.
#     """
#     # ====== YOUR CODE: ======

#     X_number_of_photons = np.asanyarray(X, np.float32) * a
#     x_poisson = np.random.poisson(X_number_of_photons)
#     x_poisson = x_poisson / a
#     x_clipped = np.clip(x_poisson, a_min=0, a_max=255)
#     Y =  np.asanyarray(x_clipped, np.uint8)
#     # ========================

#     return Y

def poisson_noisy_image(X, a):
    """
    Creates a Poisson noisy image.
    :param X: The Original image. np array of size [H x W] and of type uint8.
    :param a: number of photons scalar factor
    :return:
    Y: The noisy image. np array of size [H x W] and of type uint8.
    """
    # ====== YOUR CODE: ======
    x_float = X.astype(np.float) * a
    x_float = np.random.poisson(x_float, x_float.shape)
    x_normal = x_float / a
    y = np.clip(x_normal, 0, 255)
    Y = y.astype(np.uint8)
    # ========================
    return Y

def denoise_by_l2(Y, X, num_iter, lambda_reg):
    """
    L2 image denoising.
    :param Y: The noisy image. np array of size [H x W]
    :param X: The Original image. np array of size [H x W]
    :param num_iter: the number of iterations for the algorithm perform
    :param lambda_reg: the regularization parameter
    :return:
    Xout: The restored image. np array of size [H x W]
    Err1: The error between Xk at every iteration and Y.
    np array of size [num_iter]
    Err2: The error between Xk at every iteration and X.
    np array of size [num_iter]
    """
    # ====== YOUR CODE: ======
    Err1 = np.zeros(num_iter)
    Err2 = np.zeros(num_iter)

    img_shape = Y.shape
    
    D_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    Y = Y.flatten('F')
    X = X.flatten('F')
    X_a = Y
    for i in range(num_iter):
        G = (X_a + lambda_reg * conv_d(conv_d(X_a, D_kernel, img_shape), D_kernel.T, img_shape)) - Y
        u = (G.T @ G) / (G.T @ (G + lambda_reg * conv_d(conv_d(G, D_kernel, img_shape), D_kernel.T, img_shape)))
        X_a = X_a - u * G

        # Err1[i] = np.linalg.norm(X_a - Y) + lambda_reg*np.linalg.norm(conv_d(X_a, D_kernel, img_shape))
        t = ((X_a - Y).T) @ (X_a - Y)
        Err1[i] = (t + lambda_reg * (((conv_d(X_a, D_kernel, img_shape)).T) @ (conv_d(X_a, D_kernel, img_shape))))
        # Err2[i] = np.linalg.norm(X_a - X)
        Err2[i] = ((X_a - X).T) @ (X_a - X)

        

    Xout = np.reshape(X_a, img_shape, order='F')
    # ========================

    return Xout, Err1, Err2

def conv_d(X, D, shape):
    np.reshape(X, shape, order='F')
    res = cv2.filter2D(src=X, ddepth=cv2.CV_32F, kernel=D, borderType=cv2.BORDER_CONSTANT)
    res = res.flatten('F')
    return res

def denoise_by_TV(Y, X, num_iter, lambda_reg, epsilon0):
    """
    TV image denoising.
    :param Y: The noisy image. np array of size [H x W]
    :param X: The Original image. np array of size [H x W]
    :param num_iter: the number of iterations for the algorithm perform
    :param lambda_reg: the regularization parameter
    :param: epsilon0: small scalar for numerical stability
    :return:
    Xout: The restored image. np array of size [H x W]
    Err1: The error between Xk at every iteration and Y.
    np array of size [num_iter]
    Err2: The error between Xk at every iteration and X.
    np array of size [num_iter]
    """
    # ====== YOUR CODE: ======
    Err1 = np.zeros(num_iter)
    Err2 = np.zeros(num_iter)

    X_a = Y
    for i in range(num_iter):
        # first_der_x = np.gradient(X_a, axis=0)
        # first_der_y = np.gradient(X_a, axis=1)
        first_der_x, first_der_y = np.gradient(X_a)

        grad_norm = np.sum(first_der_x**2) + np.sum(first_der_y**2)
        normalization = np.sqrt(grad_norm + epsilon0**2)
        sec_der_x = np.gradient(first_der_x/normalization, axis=0)
        sec_der_y = np.gradient(first_der_y/normalization, axis=1)
        # assert first_der_x != np.zeros_like(first_der_x)
        # assert first_der_y != np.zeros_like(first_der_y)
        # assert sec_der_x != np.zeros_like(sec_der_x)
        # assert sec_der_y != np.zeros_like(sec_der_y)
        # grad_norm = TV(X_a)**2
        # laplace  = lambda_reg * (sec_der_x + sec_der_y)
        field = np.array((first_der_x, first_der_y ))/ normalization
        # laplace  = lambda_reg * (sec_der_x + sec_der_y)
        laplace = lambda_reg * np.sum(np.gradient(field, axis=0))
        # laplace  = lambda_reg * cv2.Laplacian(X_a, ddepth=-1) / normalization
        # assert laplace != np.zeros_like(laplace)
        u_k = 2*(Y - X_a) + laplace
        # assert u_k != np.zeros_like(u_k)
        temp = X_a
        X_a = X_a + 0.5 * 150 * epsilon0 * u_k
        if(is_same(temp, X_a)):
            print("SAME")
        x_y_norm = ((X_a.flatten('F') - Y.flatten('F')).T) @ (X_a.flatten('F') - Y.flatten('F'))
        Err1[i] = x_y_norm + lambda_reg *TV(X_a)
        Err2[i] = ((X_a.flatten('F') - X.flatten('F')).T) @ (X_a.flatten('F') - X.flatten('F'))
        

    Xout = X_a
    # ========================

    return Xout, Err1, Err2
def is_same(A, B):
    h,w = A.shape
    for i in range(h):
        for j in range(w):
            if(A[i][j] != B[i][j]):
                return False
    return True

def TV(X):
    first_der_x = np.gradient(X, axis=1)
    first_der_y = np.gradient(X, axis=0)
    tv = np.sum(np.sqrt(first_der_x**2 + first_der_y**2))

    return tv

if __name__ == '__main__':
    # 3.a - Pre-processing - Creating a noisy i mage
    vid_path = "..\given_data\Flash Gordon Trailer.mp4" 
    fps = 25
    frames = video_to_frames(vid_path=vid_path, start_second= (20*fps), end_second= (21*fps))
    img = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Color image")
    plt.axis('off')
    plt.show()
    red_img, green_img, blue_img = cv2.split(img)
    
    plt.imshow(red_img, "gray")
    plt.title("Red Channnel Image")
    plt.axis('off')
    plt.show()

    resized_img = cv2.resize(red_img, (int(red_img.shape[1]/2), int(red_img.shape[0]/2)))



    
    noisy_img = poisson_noisy_image(resized_img, 3)
    plt.imshow(noisy_img, "gray")
    plt.title("Noisy Image")
    plt.axis('off')
    plt.show()

    plt.imshow(noisy_img - resized_img, "gray")
    plt.title("Noise Image")
    plt.axis('off')
    plt.show()

# 3.b - Denoise by L2
    # lambda_reg = 0.5
    # num_iter = 50
    # Xout, Err1, Err2 = denoise_by_l2(noisy_img, resized_img, num_iter=num_iter, lambda_reg=lambda_reg)
    # plt.imshow(Xout, "gray")
    # plt.title("Restored Image (Denoise by L2)")
    # plt.axis('off')
    # plt.show()
    

    # plt.plot(range(num_iter), np.log(Err1), label="Error 1")
    # plt.plot(range(num_iter), np.log(Err2), label="Error 2")
    # plt.xlabel("number of epochs")
    # plt.ylabel("Error in log 10 scale")
    # plt.legend()
    # plt.show()
    
# 3.c - Denoise by Total Variation

    lambda_reg = 20
    num_iter = 200
    # epsilon0 = 10 ** (-6)
    epsilon0 = 2 * (10 ** (-4))
    noisy_img = np.asarray(noisy_img, dtype=np.float)
    Xout, Err1, Err2 = denoise_by_TV(noisy_img, resized_img, num_iter=num_iter, lambda_reg=lambda_reg, epsilon0 = epsilon0 )
    plt.imshow(Xout, "gray")
    plt.title("Restored Image (Denoise by Total Variation)")
    plt.axis('off')
    plt.show()

    plt.imshow(Xout- noisy_img, "gray")
    plt.show()
    

    plt.plot(range(num_iter), np.log(Err1), label="Error 1")
    plt.plot(range(num_iter), np.log(Err2), label="Error 2")
    plt.xlabel("number of epochs")
    plt.ylabel("Error in log 10 scale")
    plt.legend()
    plt.show()

