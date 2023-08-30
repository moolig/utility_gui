import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from scipy import ndimage



def aoi_in_video(aoi_im, video_im):
    pass




def video_to_binary_im(video_im):

    histr = cv.calcHist([video_im], [0], None, [256], [0, 256])
    yhat = savgol_filter(histr[:, 0], 50, 3)  # window size 50, polynomial order 3

    max_points = argrelextrema(yhat, np.greater)[0]
    max_points = [i for i in max_points if yhat[i] > video_im.size**0.5]

    video_to_edge_im(video_im)

    if len(max_points) is not 2:
        return 0

    min_points = argrelextrema(yhat, np.less)[0]
    min_point = [i for i in min_points if i > max_points[0] and i < max_points[1]][0]

    k=9
    kernel = np.ones((k, k), np.float32) / (k**2)
    dst = cv.filter2D(video_im, -1, kernel)

    ret, bw_img = cv.threshold(dst, min_point, 255, cv.THRESH_BINARY)
    return bw_img


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)

def video_to_edge_im(img):
    edges = cv.Canny(img, 100, 200)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def sift_mach(path1, path2):
    MIN_MATCH_COUNT = 10
    img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE) # queryImage
    img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE) # trainImage

    img2 = video_to_binary_im(img2)


    # Initiate SIFT detector
    sift_b = cv.SIFT_create()
    sift_v = cv.SIFT_create(sigma=9)
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


if __name__ == '__main__':
    aoi_path = r'C:\work_space\Temp\Export_29_08_23_36_19_0\4_Cad.png'
    video_path = r'C:\work_space\Temp\Export_29_08_23_36_19_0\4_Video.png'

    sift_mach(aoi_path, video_path)
    #
    # aoi_im = cv.imread(aoi_path)
    # video_im = cv.imread(video_path)
    # aoi_in_video(aoi_im, video_im)