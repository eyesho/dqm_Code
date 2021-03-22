from uiautomator import device as d
import cv2 as cv
import cv2
from PIL import Image, ImageDraw
import imutils
import numpy as np

# 16 x 16

import math


def draw_grid(height, width):
    sideline = max(height, width)
    while (sideline - 17) % 16 != 0:
        sideline += 1
    gap = (sideline - 17) // 16
    print(gap)
    grid_image = np.zeros((sideline, sideline), np.uint8)
    for i in range(1, 17):
        for j in range(1, 17):
            cv2.rectangle(grid_image, (i * gap, j * gap), (i * gap + gap, j * gap + gap), 255, 1)
    # cv2.imshow("", grid_image)
    # cv2.waitKey(111)
    return grid_image, sideline , gap


def rotate(image, angle, center=None, scale=1.0, invert=False):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if invert:
        flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    else:
        flags = None
    rotated = cv2.warpAffine(image, M, (w, h), flags=flags)
    return rotated


def conv(target_img, kernel_img):  # parallelable
    max_sum = 0
    parameter = []
    center = (w / 2, h / 2)
    result_final = np.zeros_like(kernel_img)
    for i in np.arange(-50, 50, 5):
        for j in np.arange(-50, 50, 5):
            for a in range(-4, 4):  # angle
                for s in np.arange(0.990, 1, 0.01):  # scale
                    x, y = center[0] + i, center[1] + j
                    kernel = rotate(kernel_img, a, (x, y), s)
                    result = cv2.bitwise_and(target_img, kernel) #DFT?
                    bit_sum = np.sum(result)
                    # print(result)
                    if bit_sum > max_sum:
                        result_final = result
                        max_sum = bit_sum
                        parameter.append([x, y, a, s])
                        print(parameter[-1], i, j)
                        # cv.imshow("", kernel)
                        # cv.imshow("s", target_img)
                        # cv.imshow("result", result)
                        # cv.imshow("dsad", kernel_img)
                        # cv.waitKey(1)
    print(len(parameter))
    return parameter[-1], result_final


def avg_pooling(img_, gap):
    pad_img = np.zeros((18, 18), np.uint8)
    pool_img = np.zeros((16, 16), np.uint8)
    avg_list = []
    offset = 1
    for i in range(1, 17):
        for j in range(1, 17):
            roi = img_[i * gap+offset:(i+1)*gap-offset, j * gap+offset:(j+1)*gap-offset]
            average_value = np.average(roi)
            pool_img[i-1][j-1] = int(average_value)
            avg_list.append(average_value)

    # length = np.array(avg_list).max()-np.array(avg_list).min()
    # print(img_dm_code,length,length/2)
    max_val = np.array(avg_list).max()

    cv2.rectangle(pad_img, (0, 0), (17, 17), max_val, 1)
    pad_img[1:17, 1:17] = pool_img

    # for i in range(16):
    #     for j in range(16):
    #         roi = img_[i +offset:(i+1)-offset, j +offset:(j+1)-offset]
    #         average_value = img_dm_code[i][j]
    #         if average_value < length/3:
    #             img_dm_code[i][j] = 0
    #         elif average_value > 2*length / 3:
    #             img_dm_code[i][j] = 255


    # img_dm_code = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 9, 2)
    return np.array(pad_img)


def get_rid_of_unclosed_line(im, gap):
    im[im == 1] = 0

    for i in range(1, 18):  # the last row and col didnt included if it's 17, i for horizontal
        for j in range(1, 18):  # cross shape
            d, r, u, l = 0, 0, 0, 0
            counter = 0
            lined = im[j*gap:(j+1)*gap, i*gap:i*gap+1]
            liner = im[j * gap:j * gap+1, i * gap:(i + 1) * gap]
            lineu = im[(j-1)*gap:j*gap, i*gap:i*gap+1]
            linel = im[j * gap:j * gap + 1, (i-1) * gap:(i) * gap]
            if np.all((lined == 255)):
                d = 1
                counter += 1
            if np.all((liner == 255)):
                r = 1
                counter += 1
            if np.all((lineu == 255)):
                u = 1
                counter += 1
            if np.all((linel == 255)):
                l = 1
                counter += 1
            if counter == 1:
                if d == 1 :
                    im[j * gap:(j + 1) * gap, i * gap:i * gap + 1] = 0
                if r == 1 :
                    im[j * gap:j * gap+1, i * gap:(i + 1) * gap] = 0
                if u == 1 :
                    im[(j-1)*gap+1:j*gap+1, i*gap:i*gap+1] = 0  # ?
                if l == 1:
                    im[j * gap:j * gap + 1, (i-1) * gap:(i) * gap] = 0

    # cv2.imshow('show', im)
    # cv2.waitKey()

    return im


def filter(img,pool_img,adt_,gap):
    thershold_gap_offset = 15
    # return img
    # print(img[0: 25,0:25])
    img[img > 1] = 1  # can be adjusted ?

    for i in range(1, 18):  # the last row and col didnt included if it's 17, i for horizontal
        for j in range(1, 18):

            linej = img[j*gap:(j+1)*gap, i*gap:i*gap+1]
            # linej_1 =  adt_[j*gap:(j+1)*gap, i*gap:i*gap+1]

            index = np.where(linej == 0)[0]
            if len(index) <= 0:
                cv2.line(img, (i * gap, j * gap), (i * gap, (j + 1) * gap), 255, 1)  # vertical
            # pl = pool_img[j, i - 1]
            # pr = pool_img[j, i]
            # sub_gap = max(pl, pr) - min(pl, pr)
            #
            # print(sub_gap, pl, pr)
            # pool_img[j, i-1] = 0
            # pool_img[j, i] = 255
            if 0 < len(index) <= gap//4:

                pl = pool_img[j, i-1]
                pr = pool_img[j, i]
                sub_gap = max(pl, pr) - min(pl,pr)

                print(sub_gap,pl,pr)
                if len(index) == gap//8  and sub_gap > thershold_gap_offset/2:
                    cv2.line(img, (i*gap, j*gap), (i*gap, (j+1)*gap), 255, 1)

                if sub_gap > thershold_gap_offset:
                    cv2.line(img, (i*gap, j*gap), (i*gap, (j+1)*gap), 255, 1)

                # if connected_points(adt_,(i*gap, j*gap),(i*gap, (j+1)*gap), gap,'v'):
                #     cv2.line(img, (i*gap, j*gap), (i*gap, (j+1)*gap), 255, 1)

            linei = img[j * gap:j*gap+1, i * gap:(i + 1) * gap]
            index = np.where(linei == 0)[0]
            if len(index) <= 0:
                cv2.line(img, (i * gap, j * gap), ((i+1) * gap, j * gap), 255, 1)

            if 0 < len(index) <= gap//4:
                pu = pool_img[j-1, i]
                pd = pool_img[j, i]
                sub_gap = max(pu, pd) - min(pd, pu)

                print(sub_gap, pu, pd)
                if len(index) == gap//8 and sub_gap > thershold_gap_offset/2:
                    cv2.line(img, (i * gap, j * gap), ((i + 1) * gap, j * gap), 255, 1)

                if sub_gap > thershold_gap_offset or len(index) == gap//8:
                    cv2.line(img, (i * gap, j * gap), ((i + 1) * gap, j * gap), 255, 1)
                # if connected_points(adt_,(i * gap, j * gap),((i + 1) * gap, j * gap), gap,'h'):
                #     cv2.line(img, (i * gap, j * gap), ((i + 1) * gap, j * gap), 255, 1)

            cv.imshow("rsesult", img)
            cv.imshow("78", pool_img)
            # cv.waitKey(10000)
    return img

def bool_func(bool):
    return not bool

def grid_walker(image,gap): # maze shape

    dqm_code = np.zeros((16, 16), np.uint8)
    two_dim_maze = []
    for i in range(1, 18):  # the last row and col didnt included if it's 17, i for horizontal
        tem = []
        for j in range(1, 18):
            d, r, u, l = 1, 1, 1, 1

            linel = image[j*gap:(j+1)*gap, i*gap:i*gap+1]    # d -> l r -> u
            lineu = image[j * gap:j * gap+1, i * gap:(i + 1) * gap]
            liner = image[j*gap:(j+1)*gap, (i+1)*gap:(i+1)*gap+1]
            lined = image[(j+1) * gap:(j+1) * gap + 1, i * gap:(i + 1) * gap]

            if np.all((lined == 255)):
                d = 0
            if np.all((liner == 255)):
                r = 0
            if np.all((lineu == 255)):
                u = 0
            if np.all((linel == 255)):
                l = 0
            tem.append([u,d,l,r])
        two_dim_maze.append(tem)
    binary = 0
    for j in range(16):
        for i in range(16):
            if i == 0 :
                if two_dim_maze[j][i][0] == 0:
                    binary = 0
                else:
                    binary = 1
                dqm_code[i][j] = binary
            else:
                if two_dim_maze[j][i-1][1] == 1:
                    binary = binary
                else:
                    binary = bool_func(binary)
                dqm_code[i][j] = binary

    # print()
    dqm_code[dqm_code == 1] = 255
    return dqm_code


if __name__ == "__main__":
    img = cv.imread('sss.jpeg', 0)

    img = cv.GaussianBlur(img, (5, 5), 0)
    canny_img = cv.Canny(img, 60, 140)
    # canny_img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 2)

    kernel = np.ones((13, 13), np.uint8)
    dilation = cv2.dilate(canny_img, kernel, iterations=1)
    dilation = cv2.erode(dilation, kernel, iterations=1)

    adt = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV, 11, 3)
    # adt = cv.bilateralFilter(adt, 9, 75, 75)
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w) / h
            xmin, ymin, xmax, ymax = x, y, x + w, y + h
            x_center, y_center = (xmax - xmin) // 2, (ymax - ymin) // 2

            if 0.95 <= aspectRatio < 1.05 and w * h > 10000:

                kernel, sideline, gap = draw_grid(h, w)
                sideline = sideline
                canny_image_roi = np.zeros((sideline, sideline), np.uint8)
                ymin, xmin = ymin-gap//2, xmin-gap//2
                ymax, xmax = ymin + sideline, xmin + sideline
                canny_image_roi[:, :] = adt[ymin:ymax, xmin:xmax]

                # cv2.waitKey(4000)

                (x, y, a, s), cov_result = conv(kernel, canny_image_roi)
                offset = 0
                img_map = rotate(img[ymin-offset:ymax-offset, xmin-offset:xmax-offset], a, (x, y), s)
                adt = rotate(canny_img[ymin-offset:ymax-offset, xmin-offset:xmax-offset], a, (x, y), s)

                pool_img = avg_pooling(img_map, gap)
                filtered_img = filter(cov_result, pool_img, canny_image_roi, gap)
                closed_img = get_rid_of_unclosed_line(filtered_img,gap)
                dpm_code = grid_walker(closed_img,gap)
                cv2.imshow("Imagssse", pool_img)
                cv2.imshow("final", dpm_code)
                cv2.imshow("Imagse", img[ymin:ymax, xmin:xmax])

                dst = cv.addWeighted(kernel, 0.08, img_map, 0.92, 0.0)

                cv2.imshow("Imagssdse", dst)
                cv2.imwrite("ss.png",dpm_code)
    # cv2.imshow("Imagssse", canny_img)
    cv2.waitKey(0)

# cv2.rectangle(canny_img, (x, y), (x+w, y+h), (255, 255, 255), 1)




