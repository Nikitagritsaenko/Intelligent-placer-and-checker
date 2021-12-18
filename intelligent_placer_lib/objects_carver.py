from __future__ import print_function

import os
from collections import Counter
from os import listdir
from os.path import isfile, join
from random import sample
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from intelligent_placer_lib.placer import get_rect, get_ideal_polygon

# 2850x4000

predefined_convex_hulls = [
    [get_rect([45, 95]), "плеер"],
    [get_rect([45, 95]), "плеер"],
    [get_rect([45, 95]), "плеер"],
    [[[117, 81], [106, 101], [95, 103], [93, 110], [75, 118], [67, 110], [20, 56], [15, 21], [33, 7], [65, 15], [86, 44], [104, 66]], "мышь"],
    [[[117, 81], [106, 101], [95, 103], [93, 110], [75, 118], [67, 110], [20, 56], [15, 21], [33, 7], [65, 15], [86, 44], [104, 66]], "мышь"],
    [[[117, 81], [106, 101], [95, 103], [93, 110], [75, 118], [67, 110], [20, 56], [15, 21], [33, 7], [65, 15], [86, 44], [104, 66]], "мышь"],
    [get_rect([55, 85]), "проездной"],
    [get_rect([55, 85]), "проездной"],
    [get_rect([55, 85]), "проездной"],
    [get_ideal_polygon(18, 40), "масло для губ"],
    [get_ideal_polygon(18, 40), "масло для губ"],
    [get_ideal_polygon(18, 40), "масло для губ"],
    [get_rect([68, 18]), "флешка"],
    [get_rect([68, 18]), "флешка"],
    [get_rect([68, 18]), "флешка"],
    [[[42, 75], [21, 68], [7, 55], [4, 42], [4, 28], [13, 12], [23, 4], [45, 6], [56, 2], [65, 10], [85, 31], [76, 40], [60, 60]], "рулетка"],
    [[[42, 75], [21, 68], [7, 55], [4, 42], [4, 28], [13, 12], [23, 4], [45, 6], [56, 2], [65, 10], [85, 31], [76, 40], [60, 60]], "рулетка"],
    [get_ideal_polygon(30, 40), "йо-йо"],
    [get_ideal_polygon(30, 40), "йо-йо"],
    [get_ideal_polygon(30, 40), "йо-йо"],
    [get_ideal_polygon(30, 40), "йо-йо"],
    [[[18, 45], [11, 29], [11, 17], [17, 7], [30, 5], [41, 13], [41, 30], [31, 46]], "точилка"],
    [[[18, 45], [11, 29], [11, 17], [17, 7], [30, 5], [41, 13], [41, 30], [31, 46]], "точилка"],
    [[[18, 45], [11, 29], [11, 17], [17, 7], [30, 5], [41, 13], [41, 30], [31, 46]], "точилка"],
    [[[18, 45], [11, 29], [11, 17], [17, 7], [30, 5], [41, 13], [41, 30], [31, 46]], "точилка"],
    [[[18, 45], [11, 29], [11, 17], [17, 7], [30, 5], [41, 13], [41, 30], [31, 46]], "точилка"],
    [[[18, 45], [11, 29], [11, 17], [17, 7], [30, 5], [41, 13], [41, 30], [31, 46]], "точилка"],
    [[[18, 45], [11, 29], [11, 17], [17, 7], [30, 5], [41, 13], [41, 30], [31, 46]], "точилка"],
    [[[18, 45], [11, 29], [11, 17], [17, 7], [30, 5], [41, 13], [41, 30], [31, 46]], "точилка"],
    [get_rect([63, 40]), "футляр для наушников"],
    [get_rect([63, 40]), "футляр для наушников"],
    [get_rect([63, 40]), "футляр для наушников"],
    [get_rect([63, 40]), "футляр для наушников"]
]


def make_pointList_from_2darray(arr):
    points = []
    for p in arr:
        points.append([p[0], p[1]])
    return points


def get_cluster_with_most_points(X):
    scaled_X = StandardScaler().fit_transform(X)
    # scaled_X = X
    clustering = DBSCAN(eps=0.3, min_samples=2).fit(scaled_X)
    # print(clustering.labels_)
    points = demo_plot(clustering, X)

    return points


def get_largest_cluster_label(cluster_labels):
    c = Counter(cluster_labels).most_common()[0]
    return c[0]


def get_point_array_from_hull(hull):
    lst = []
    for arr_point in hull:
        lst.append(arr_point[0])
    return np.array(lst)


def demo_plot(db, X):
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    largest_cluster_label = get_largest_cluster_label(labels)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    # unique_labels = set(largest_cluster_label)
    # colors = [plt.cm.Spectral(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    # if k == -1:
    #     # Black used for noise.
    #     col = [0, 0, 0, 1]
    col = [1, 0, 0, 1]
    class_member_mask = (labels == largest_cluster_label)

    xy = X[class_member_mask & core_samples_mask]

    hull = cv2.convexHull(xy)

    hull_points = get_point_array_from_hull(hull)

    plt.plot(hull_points[:, 0], hull_points[:, 1], 'o', markerfacecolor=tuple(col),
             markersize=5)

    # xy = X[class_member_mask & ~core_samples_mask]
    # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #          markeredgecolor='b', markersize=3)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    return hull_points


def convex_hull(src):
    image = src.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    all_points = []

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x, y] != 255:
                all_points.append(np.array([y, -x]))

    print(len(all_points))

    part_of_points = sample(all_points, int(len(all_points) * 0.01))

    points = get_cluster_with_most_points(np.array(part_of_points))

    return points


def thresh_callback(src, threshold, verbose=False):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3, 3))
    height = src.shape[0]
    width = src.shape[1]
    max_thresh = 255
    # cv2.createTrackbar('Canny thresh:', "source_window", threshold, max_thresh, thresh_callback)

    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(canny_output, cv2.MORPH_CLOSE, kernel)

    try:
        _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        # MChepulis: что-то с библиотекой cv2. Чтобы не портить старый код сделал заглушку
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = filter(lambda cont: cv2.arcLength(cont, False) > 10, contours)
    num_cnt = 0
    for i, c in enumerate(contours):
        if (cv2.arcLength(c, False) < 500):
            continue
        num_cnt += 1

    contours_poly = [None] * num_cnt
    boundRect = [None] * num_cnt
    centers = [None] * num_cnt
    radius = [None] * num_cnt
    index = -1
    for i, c in enumerate(contours):
        if (cv2.arcLength(c, False) < 500):
            continue
        index += 1
        contours_poly[index] = cv2.approxPolyDP(c, 3, True)
        boundRect[index] = cv2.boundingRect(contours_poly[index])
        centers[index], radius[index] = cv2.minEnclosingCircle(contours_poly[index])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    hierarchy = hierarchy[0]
    if verbose:
        print("Общее Количество контуров", num_cnt)

    color = [(255, 44, 0),
             (254, 178, 0),
             (161, 238, 0),
             (129, 6, 168),
             (0, 165, 124),
             (18, 62, 170),
             ]
    cropped_objects = []
    counter = 0
    for i, c in enumerate(contours):
        if (cv2.arcLength(c, False) < 500):
            continue
        if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
            cv2.drawContours(drawing, contours, i, color[counter % 5], 2)
        else:
            cv2.drawContours(drawing, contours, i, (0, 255, 0), 2)

        y, x, h, w, = boundRect[counter]
        if (w < 20 or h < 20):
            continue
        cropped_objects.append(
            src[(int)(get_min(x, 0)):(int)(get_max(x + w, height)), (int)(get_min(y, 0)):(int)(get_max(y + h, width))])
        # cropped_objects.append(get_rect([1,1]))
        counter += 1
    if verbose:
        print("Длинное Количество контуров", counter)

    # x,y,w,h 

    return drawing, cropped_objects


def get_min(x, bound):
    if (x > bound + 10):
        return x - 10
    return bound


def get_max(x, bound):
    if (x < bound - 10):
        return x + 10
    return bound


def read_image(filename, directory):
    path = os.path.join(directory, filename)
    original_image = imread(path)
    return original_image


def cut_edges(original_image, cut_threshold):
    h, w, _ = original_image.shape
    cropped_image = original_image[(int)(h * cut_threshold):(int)(h * (1 - cut_threshold)),
                    (int)(w * cut_threshold):(int)(w * (1 - cut_threshold))]
    return cropped_image


def match_points(src, dst, verbose=False):
    try:
        sift = cv2.xfeatures2d.SIFT_create(sigma=3.5)
    except:
        # MChepulis: что-то с библиотекой cv2. Чтобы не портить старый код сделал заглушку
        sift = cv2.SIFT_create(sigma=3.5)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(src, None)
    kp2, des2 = sift.detectAndCompute(dst, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if verbose and len(good_matches) > 0:
        img_matches = cv2.drawMatches(src, kp1, dst, kp2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_matches, interpolation='nearest')
        plt.show()

    return good_matches


def find_objects_on_img(path, verbose=False):
    thresh = 100
    image = read_image(path, "")

    result, cropped_objects = thresh_callback(image, thresh, verbose=verbose)

    single_item_dataset_directory = os.path.join(os.path.curdir, "intelligent_placer_lib", "ML2021Dataset")
    single_item_files = [f for f in listdir(single_item_dataset_directory) if
                         isfile(join(single_item_dataset_directory, f))]

    single_item_files.sort(key=lambda s: int(s.rstrip('.jpg')))
    best_matches_len = []
    best_matches_names = []
    best_matches_polygons = []
    for fragment in cropped_objects:
        src = cv2.cvtColor(fragment, cv2.COLOR_BGR2GRAY)
        best_match_len = 0
        best_i = 1
        for i, single_item_file in enumerate(single_item_files, 0):
            dst = cv2.imread(join(single_item_dataset_directory, single_item_file))
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            matches = match_points(src, dst, verbose=verbose)
            len_matches = len(matches)
            if len_matches > best_match_len:
                best_match_len = len_matches
                best_i = i
        best_matches_len.append(best_match_len)
        best_matches_polygons.append(predefined_convex_hulls[best_i][0])
        best_matches_names.append(predefined_convex_hulls[best_i][1])

    return best_matches_names, best_matches_polygons
