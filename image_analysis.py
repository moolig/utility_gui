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



def aoi_in_video(aoi_im, video_im):
    pass




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



def sift_mach(aoi_path, video_path):
    MIN_MATCH_COUNT = 10
    img1 = cv.imread(aoi_path, cv.IMREAD_GRAYSCALE) # queryImage
    img2 = cv.imread(video_path, cv.IMREAD_GRAYSCALE) # trainImage


    img1 = im_to_binary_im(img1, 3)
    img2 = im_to_binary_im(img2, 5)

    kernel = np.ones((5, 5), np.uint8)
    img2 = cv.dilate(img2, kernel, iterations=2)
    img2 = cv.erode(img2, kernel, iterations=2)

    # img1 = image_to_edge_im(img1)
    # img2 = image_to_edge_im(img2)


    # Initiate SIFT detector
    sift_b = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift_b.detectAndCompute(img1,None)
    kp2, des2 = sift_b.detectAndCompute(img2,None)

    # img = cv.drawKeypoints(img1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img, 'gray'), plt.show()
    #
    #
    # img_2 = cv.drawKeypoints(img2, kp2, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img_2, 'gray'), plt.show()


    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
         singlePointColor = None,
         matchesMask = matchesMask, # draw only inliers
         flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()



def test_func(path):
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    image = cv.GaussianBlur(image, (5, 5), 0)

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
    print("Otsu's algorithm implementation thresholding result: ", threshold)

    # Use a bimodal image as an input.
    # Optimal threshold value is determined automatically.
    otsu_threshold, image_result = cv.threshold(
        image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
    )
    print("Obtained threshold: ", otsu_threshold)


def test_from_chatgpt(big_image, cropped_image):
    big_image = cv.imread(big_image)
    cropped_image = cv.imread(cropped_image)

    # Convert images to grayscale
    gray_big_image = cv.cvtColor(big_image, cv.COLOR_BGR2GRAY)
    gray_cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)



    gray_cropped_image = im_to_binary_im(gray_cropped_image, 3)

    img2 = im_to_binary_im(gray_big_image, 5)
    kernel = np.ones((5, 5), np.uint8)
    img2 = cv.dilate(img2, kernel, iterations=2)
    gray_big_image = cv.erode(img2, kernel, iterations=2)



    # Initialize feature detector and descriptor
    orb = cv.ORB_create()
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray_cropped_image, None)
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    kps1, dess1 = sift.detectAndCompute(gray_cropped_image, None)




    kp2, des2 = orb.detectAndCompute(gray_big_image, None)





    img = cv.drawKeypoints(gray_cropped_image, kp1, gray_cropped_image, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img, 'gray'), plt.show()

    img = cv.drawKeypoints(gray_big_image, kp2, gray_big_image, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img, 'gray'), plt.show()

    # Initialize a brute-force matcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate transformation matrix using RANSAC
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    # Apply transformation to the cropped image
    height, width = cropped_image.shape[:2]
    aligned_cropped_image = cv.warpPerspective(cropped_image, M, (width, height))


def bruth_forth(big_image, cropped_image):
    gray_big_image = cv.imread(big_image, cv.IMREAD_GRAYSCALE)
    gray_cropped_image = cv.imread(cropped_image, cv.IMREAD_GRAYSCALE)

    w_aoi, h_aoi = gray_cropped_image.shape
    w, h = gray_big_image.shape

    cropped_bm = im_to_binary_im(gray_cropped_image, 3)
    big_bm = im_to_binary_im(gray_big_image, 5)

    kernel = np.ones((5, 5), np.uint8)
    big_bm = cv.dilate(big_bm, kernel, iterations=2)
    big_bm = cv.erode(big_bm, kernel, iterations=2)


    '''
    test revert to gray
    '''
    big_bm = gray_big_image
    cropped_bm = gray_cropped_image
    #-----------------------


    method = cv.TM_CCOEFF_NORMED

    scale = 0.9  # percent of original size
    threshold = 3
    glob_max_val = 0
    res_rotate = 0
    res_resize = 1

    resize_time = 10

    w = int(w * scale**5)
    h = int(h * scale**5)
    dim = (w, h)
    big_bm = cv.resize(big_bm, dim, interpolation=cv.INTER_AREA)

    for rsize in range(1,resize_time):
        for rotate in range(4):
            for flip in range(2):

                temp_im = cropped_bm - cropped_bm.mean()
                res1 = signal.correlate2d(big_bm, temp_im, boundary='symm', mode='same')
                res = res1



                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                max_val = max_val/(255*w_aoi*h_aoi)

                x_rec_size = min(w, int(max_loc[0] + w_aoi / 2)) - max(0, int(max_loc[0] - w_aoi / 2))
                y_rec_size = min(h, int(max_loc[1] + h_aoi / 2)) - max(0, int(max_loc[1] - h_aoi / 2))
                rec_area = x_rec_size * y_rec_size


                max_val = max_val * (rec_area / (h_aoi * w_aoi))

                if max_val > glob_max_val:
                    glob_max_val = max_val
                    res_rotate = rotate
                    res_resize = rsize
                    res_max_loc = max_loc
                    res_aoi_im = cropped_bm.copy()
                    res_video_im = big_bm.copy()
                # print(max_val)


                # loc = np.where(res >= threshold)
                # for pt in zip(*loc[::-1]):  # Switch columns and rows
                #     cv.rectangle(gray_big_image, pt, (pt[0] + w, pt[1] + h), (255), 1)

                # img3 = cv.drawMatches(big_bm, [], cropped_bm, [], [], None)
                # plt.imshow(img3, 'gray'), plt.show()

                cropped_bm = cv.flip(cropped_bm, 0)


            cropped_bm = cv.rotate(cropped_bm, cv.ROTATE_90_CLOCKWISE)
        w = int(w * scale)
        h = int(h * scale)
        dim = (w, h)
        # resize image
        big_bm = cv.resize(big_bm, dim, interpolation=cv.INTER_AREA)


    w_v, h_v = res_video_im.shape
    w, h = gray_cropped_image.shape
    # cv.rectangle(gray_big_image, res_max_loc, (res_max_loc[0] + w, res_max_loc[1] + h), (255), 1)

    cv.rectangle(res_video_im, (int(res_max_loc[0] - w / 2), int(res_max_loc[1] - h / 2)),
                 (int(res_max_loc[0] + w / 2), int(res_max_loc[1] + h / 2)), (255), 1)

    # x_rec_size = min(w_v, int(res_max_loc[0] + w / 2)) - max(0, int(res_max_loc[0] - w / 2))
    # y_rec_size = min(h_v, int(res_max_loc[1] + h / 2)) - max(0, int(res_max_loc[1] - h / 2))
    # rec_area = x_rec_size * y_rec_size

    img3 = cv.drawMatches(res_video_im, [], res_aoi_im, [], [], None)
    # glob_max_val = glob_max_val*(rec_area/(h*w))
    if glob_max_val < threshold:
        with open(r"C:\work_space\Temp\res\unfit.txt", "a") as f:
            f.write(big_image + ':' + str(glob_max_val) + '\n')
            # Reading form a file
            # plt.imshow(img3, 'gray'), plt.show()
        cv.imwrite(os.path.join(r'C:\work_space\Temp\res\unfit', os.path.basename(big_image)), img3)
    else:
        with open(r"C:\work_space\Temp\res\fit.txt", "a") as f:
            f.write(big_image + ':' + str(glob_max_val) + '\n')
            # Reading form a file
            # plt.imshow(img3, 'gray'), plt.show()
        cv.imwrite(os.path.join(r'C:\work_space\Temp\res\fit', os.path.basename(big_image)), img3)


        print(glob_max_val)
        print(big_image)




if __name__ == '__main__':

    dir_path = r'C:\work_space\temp\cims_im\image'
    aoi_ims = [f for f in os.listdir(dir_path) if '_Image.png' in f]

    #for i in range(0,1126):#(0,20458):
    for aoi_im in aoi_ims:
        try:
            # aoi_path = fr'C:\work_space\Temp\Export_29_08_23_36_19_0\{i}_Image.png'
            # video_path = fr'C:\work_space\Temp\Export_29_08_23_36_19_0\{i}_Video.png'
            aoi_path = os.path.join(dir_path, aoi_im)
            video_path = os.path.join(dir_path, aoi_im.replace('_Image', '_Video'))
            bruth_forth(video_path, aoi_path)
        except:
            with open(r"C:\work_space\Temp\res\temp.txt", "a") as f:
                # Reading form a file
                f.write('---' + aoi_path + '\n')
