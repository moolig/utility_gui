import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from scipy import ndimage

from scipy import signal
from scipy import datasets
from numpy.fft import fft2, ifft2

import os



def find_circle_radius_in_image(im_path, min_radius=10, max_radius=200):
    gray = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
    h, w = gray.shape
    # gray = im_to_binary_im(gray, 3)
    #
    #
    # kernel = np.ones((5, 5), np.uint8)
    # gray = cv.dilate(gray, kernel, iterations=2)
    # gray = cv.erode(gray, kernel, iterations=2)

    # Apply Gaussian blur to reduce noise and improve circle detection
    blurred = cv.GaussianBlur(gray, (9, 9), 4)

    # Use the Hough Circle Transform to detect circles
    circles1 = cv.HoughCircles(
        blurred, cv.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=50, minRadius=min_radius, maxRadius=max_radius
    )

    circles = cv.HoughCircles(
        blurred, cv.HOUGH_GRADIENT_ALT, dp=1.5, minDist=20,
        param1=50, param2=0.8, minRadius=int(h/20), maxRadius=max_radius
    )

    circles3 = cv.HoughCircles(
        blurred, cv.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=50, minRadius=40, maxRadius=max_radius
    )

    if circles is not None:
        # Convert circle coordinates to integers
        circles = np.uint16(np.around(circles))

        cimg = cv.cvtColor(blurred, cv.COLOR_GRAY2BGR)
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        # Initialize a list to store circle radii
        radii = []

        # Loop through detected circles and extract radii
        for circle in circles[0, :]:
            center_x, center_y, radius = circle[0], circle[1], circle[2]
            radii.append(radius)

        return radii
    else:
        return []




def im_to_binary_im(video_im, k):
    image = cv.GaussianBlur(video_im, (k, k), 0)
    bins_num = 256
    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required

    hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    # print("Otsu's algorithm implementation thresholding result: ", threshold)

    # Use a bimodal image as an input.
    # Optimal threshold value is determined automatically.
    otsu_threshold, image_result = cv.threshold(
        image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
    )

    return image_result



def image_to_edge_im(img):
    edges = cv.Canny(img, 100, 200)
    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    return edges



def show_im(img, aoi, max_loc, res, w, h):

    top_left = max_loc
    copy = img.copy()

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(copy, top_left, bottom_right, 255, 2)

    plt.subplot(131), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(copy, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(aoi, cmap='gray')
    plt.title('aoi'), plt.xticks([]), plt.yticks([])


    plt.show()


def brute_force(big_image, cropped_image, res_root=None, threshold=3.0, start_scale=1):
    """
    Perform a 'brute_force' operation on images.

    This function takes a 'big_image' and a 'cropped_image' as input and performs a 'bruth forth' operation on them.

    Parameters:
    - big_image (str): The path to the video image file representing the big image.
    - cropped_image (str): The path to the aoi image file representing the cropped image.
    - res_root (str): The root directory where the result will be saved. None if try to find best scale.

    - threshold (float, optional): The threshold value for a specific operation (default is 3).

    - start_scale

    Returns:
    - glob_max_val: best

    Note:
    - The 'bruth forth' operation is a custom image processing technique applied to 'big_image' and 'cropped_image' to generate a result.
    - You can adjust the 'threshold' and 'resize_time' parameters to control the behavior of the operation.
    - The result of the operation will be saved in the 'res_root' directory with a unique filename.

    Example usage:
    bruth_forth("big_image.jpg", "cropped_image.jpg", "results/", scale=0.5, threshold=5, resize_time=2)
    """
    rgb_video_im = cv.imread(big_image)
    # rgb_video_im = cv.bilateralFilter(rgb_video_im, 9, 75, 75)

    gray_big_image = cv.cvtColor(rgb_video_im, cv.COLOR_BGR2GRAY)
    gray_aoi_image = cv.imread(cropped_image, cv.IMREAD_GRAYSCALE)




    w_aoi, h_aoi = gray_aoi_image.shape
    w, h = gray_big_image.shape
    orig_w_v, orig_h_v = w, h

    aoi_bm = im_to_binary_im(gray_aoi_image, 3)
    big_bm = im_to_binary_im(gray_big_image, 5)

    kernel = np.ones((5, 5), np.uint8)
    big_bm = cv.dilate(big_bm, kernel, iterations=2)
    big_bm = cv.erode(big_bm, kernel, iterations=2)




    glob_max_val = 0

    w = int(w * start_scale)
    h = int(h * start_scale)
    # big_bm = cv.resize(big_bm, (w, h), interpolation=cv.INTER_AREA)
    gray_big_image = cv.resize(gray_big_image, (w, h), interpolation=cv.INTER_AREA)
    gray_big_image = cv.copyMakeBorder(gray_big_image, int(h_aoi / 4), int(h_aoi / 4), int(w_aoi / 4), int(w_aoi / 4),
                                                                                 cv.BORDER_REPLICATE)

    big_bm = cv.resize(big_bm, (w, h), interpolation=cv.INTER_AREA)
    big_bm = cv.copyMakeBorder(big_bm, int(h_aoi / 4), int(h_aoi / 4), int(w_aoi / 4), int(w_aoi / 4),
                                                                                 cv.BORDER_REPLICATE)

    # gray_big_image = cv.rotate(gray_big_image, cv.ROTATE_180)


    for rotate in range(4):
        for flip in range(2):

            # cv.imwrite(r'C:\work_space\Temp\por_utc0380_3f_from_atsiq_050923_res\test\aoi2.jpg', gray_aoi_image)
            # cv.imwrite(r'C:\work_space\Temp\por_utc0380_3f_from_atsiq_050923_res\test\video2.jpg', gray_big_image)
            # res = cv.matchTemplate(gray_big_image, gray_aoi_image, cv.TM_CCOEFF)
            res = cv.matchTemplate(big_bm, aoi_bm, cv.TM_CCOEFF)

            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            max_val = max_val/(255*w_aoi*h_aoi)

            if max_val > glob_max_val:
                glob_max_val = max_val
                res_rgb_video_im = rgb_video_im
                res_max_loc = max_loc
                res_video_im = gray_big_image.copy()

            # print(max_val, ' , ', max_loc)
            # show_im(gray_big_image, gray_aoi_image, max_loc, res, w_aoi, h_aoi)
            # show_im(big_bm, aoi_bm, max_loc, res, w_aoi, h_aoi)

            big_bm = cv.flip(big_bm, 0)
            gray_big_image = cv.flip(gray_big_image, 0)
            rgb_video_im = cv.flip(rgb_video_im, 0)

        big_bm = cv.rotate(big_bm, cv.ROTATE_90_CLOCKWISE)
        gray_big_image = cv.rotate(gray_big_image, cv.ROTATE_90_CLOCKWISE)
        rgb_video_im = cv.rotate(rgb_video_im, cv.ROTATE_90_CLOCKWISE)




    if res_root is not None:
        w_v, h_v = res_video_im.shape

        # cv.rectangle(res_video_im, (int(res_max_loc[0] - w_aoi / 2), int(res_max_loc[1] - h_aoi / 2)),
        #              (int(res_max_loc[0] + w_aoi / 2), int(res_max_loc[1] + h_aoi / 2)), (255), 1)

        top_left = res_max_loc
        bottom_right = (top_left[0] + w_aoi, top_left[1] + h_aoi)

        cv.rectangle(res_video_im, top_left, bottom_right, (255), 1)

        img3 = cv.drawMatches(res_video_im, [], gray_aoi_image, [], [], None)
        img4 = cv.drawMatches(img3, [], res_video_im[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], [], [], None)

        # cv.rectangle(res_rgb_video_im, (int((res_max_loc[0] - w / 2)*(orig_w_v/w_v)), int((res_max_loc[1] - h / 2)*(orig_h_v/h_v))),
        #              (int((res_max_loc[0] + w / 2)*(orig_w_v/w_v)), int((res_max_loc[1] + h / 2)*(orig_h_v/h_v))), (255,0,0), 1)

        top_left = (int(top_left[0]*(orig_h_v/h_v)), int(top_left[1]*(orig_h_v/h_v)))
        bottom_right = (top_left[0] + w_aoi, top_left[1] + h_aoi)

        cv.rectangle(res_rgb_video_im, top_left, bottom_right, (255,0,0), 1)

        if glob_max_val < threshold:
            str_results = 'unfit'
        else:
            str_results = 'fit'

        with open(os.path.join(res_root, f"{str_results}.txt"), "a") as f:
            f.write(big_image + ':' + str(glob_max_val) + '\n')
            # Reading form a file
            # plt.imshow(img3, 'gray'), plt.show()
        cv.imwrite(os.path.join(res_root, f'{str_results}', os.path.basename(big_image)), img4)
        cv.imwrite(os.path.join(res_root, f'{str_results}_im', os.path.basename(big_image)), res_rgb_video_im)
        cv.imwrite(os.path.join(res_root, f'{str_results}_im', os.path.basename(cropped_image)), gray_aoi_image)



    return glob_max_val




def get_best_magnitude_from_im(aoi_path):

    video_path = aoi_path.replace('_Image', '_Video')

    circle_rad_list_video = find_circle_radius_in_image(video_path)
    circle_rad_list_aoi = find_circle_radius_in_image(aoi_path)

    if len(circle_rad_list_aoi) and len(circle_rad_list_video):
        average = sum(circle_rad_list_video) / len(circle_rad_list_video)
        aoi = max(circle_rad_list_aoi)
        return aoi/average
    return 1







def run_bruth_forth_on_dir(dir_path, test_magnitude_im, threshold=5):

    start_scale = get_best_magnitude_from_im(os.path.join(dir_path, test_magnitude_im))


    os.makedirs(fr'{dir_path}_res', exist_ok=True)
    os.makedirs(os.path.join(fr'{dir_path}_res', 'fit'), exist_ok=True)
    os.makedirs(os.path.join(fr'{dir_path}_res', 'unfit'), exist_ok=True)
    os.makedirs(os.path.join(fr'{dir_path}_res', 'fit_im'), exist_ok=True)
    os.makedirs(os.path.join(fr'{dir_path}_res', 'unfit_im'), exist_ok=True)

    aoi_ims = [f for f in os.listdir(dir_path) if '_Image.png' in f]


    for idx, aoi_im in enumerate(aoi_ims):


        aoi_path = os.path.join(dir_path, aoi_im)
        video_path = os.path.join(dir_path, aoi_im.replace('_Image', '_Video'))
        glob_max_val = brute_force(video_path, aoi_path, fr'{dir_path}_res', threshold=threshold, start_scale=start_scale)
        print(f'{aoi_path}: {glob_max_val}')




if __name__ == '__main__':


    test_magnitude_im = r'325_Image.png'
    dir_path = r'C:\work_space\Temp\por_utc0380_3f_from_atsiq_050923'

    run_bruth_forth_on_dir(dir_path, test_magnitude_im, 25)




    from matplotlib import pyplot as plt


    aoi = cv.imread(r'C:\work_space\Temp\por_utc0380_3f_from_atsiq_050923\126_Image.png', cv.IMREAD_GRAYSCALE)
    sh, sw = aoi.shape
    video = cv.imread(r'C:\work_space\Temp\por_utc0380_3f_from_atsiq_050923\126_Video.png', cv.IMREAD_GRAYSCALE)

    glob_max_val = brute_force( r'C:\work_space\Temp\por_utc0380_3f_from_atsiq_050923\153_Video.png', r'C:\work_space\Temp\por_utc0380_3f_from_atsiq_050923\153_Image.png', fr'C:\work_space\Temp\por_utc0380_3f_from_atsiq_050923_res\test', threshold=3, start_scale=0.71)
    h, w = video.shape
    f = 0.71
    video = cv.resize(video, (int(f*w), int(f*h)), interpolation=cv.INTER_AREA)


    video = cv.rotate(video, cv.ROTATE_180)
    # video = cv.flip(video, 0)

    ped_video_reflect = cv.copyMakeBorder(video, int(sh / 4), int(sh / 4), int(sw / 4), int(sw / 4),
                                          cv.BORDER_REPLICATE)

    lh, lw = ped_video_reflect.shape

    ped_aoi = cv.copyMakeBorder(aoi, int((lh-sh) / 2), int((lh-sh) / 2), int((lh-sh) / 2), int((lh-sh) / 2),
                                          cv.BORDER_CONSTANT,value=0)

    cv.imwrite(r'C:\work_space\Temp\test\ped_video_resize_gray.png', ped_video_reflect)
    print('end')

    w, h = aoi.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF']#, 'cv.TM_CCOEFF_NORMED','cv.TM_CCORR_NORMED']

    for meth in methods:
        img = ped_video_reflect.copy()
        method = eval(meth)
        # Apply template Matching

        cv.imwrite(r'C:\work_space\Temp\por_utc0380_3f_from_atsiq_050923_res\test\aoi1.jpg', aoi)
        cv.imwrite(r'C:\work_space\Temp\por_utc0380_3f_from_atsiq_050923_res\test\video1.jpg', img)


        res = cv.matchTemplate(img, aoi, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)



        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
            print(min_val, ", ", top_left)
        else:
            top_left = max_loc
            print(max_val, ", ", top_left)
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img, top_left, bottom_right, 255, 2)
        plt.subplot(131), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

        plt.subplot(133), plt.imshow(ped_aoi, cmap='gray')
        plt.title('aoi'), plt.xticks([]), plt.yticks([])

        plt.suptitle(meth)
        plt.show()




    # brute_force(video_path, aoi_path, fr'{dir_path}_res')



