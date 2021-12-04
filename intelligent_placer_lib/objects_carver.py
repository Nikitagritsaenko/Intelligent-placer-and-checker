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

from placer import placer, get_rect, get_ideal_polygon

# 2850x4000

predefined_convex_hulls = [
    get_rect([600,1300]),
    get_rect([600,1300]),
    get_rect([600,1300]),
    [[400,0],[0,500],[250,1700],[1150,1550],[900,220]],
    [[400,0],[0,500],[250,1700],[1150,1550],[900,220]],
    [[400,0],[0,500],[250,1700],[1150,1550],[900,220]],
    get_rect([730,1170]),
    get_rect([730,1170]),
    get_rect([730,1170]),
    get_ideal_polygon(265, 8),
    get_ideal_polygon(265, 8),
    get_ideal_polygon(265, 8),
    get_rect([1050,300]),
    get_rect([1050,300]),
    get_rect([1050,300]),
    [[0,600],[50,290],[50,850],[270,1370],[900,1160],[1170,600]],
    [[0,600],[50,290],[50,850],[270,1370],[900,1160],[1170,600]],
    get_ideal_polygon(500, 8),
    get_ideal_polygon(500, 8),
    get_ideal_polygon(500, 8),
    get_ideal_polygon(500, 8),
    [[250,0],[50,50],[0,200],[180,630],[380,640],[530,250],[470,50]],
    [[250, 0], [50, 50], [0, 200], [180, 630], [380, 640], [530, 250], [470, 50]],
    [[250, 0], [50, 50], [0, 200], [180, 630], [380, 640], [530, 250], [470, 50]],
    [[250, 0], [50, 50], [0, 200], [180, 630], [380, 640], [530, 250], [470, 50]],
    [[250, 0], [50, 50], [0, 200], [180, 630], [380, 640], [530, 250], [470, 50]],
    [[250, 0], [50, 50], [0, 200], [180, 630], [380, 640], [530, 250], [470, 50]],
    [[250, 0], [50, 50], [0, 200], [180, 630], [380, 640], [530, 250], [470, 50]],
    [[250, 0], [50, 50], [0, 200], [180, 630], [380, 640], [530, 250], [470, 50]],
    get_rect([800,500]),
    get_rect([800,500]),
    get_rect([800,500]),
    get_rect([800,500]),
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
    #print(clustering.labels_)
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
    
    
    
    # convex_hull_points = cv2.convexHull(np.array(points))
    
    #     result_image = image.copy()
    #     for x in range(result_image.shape[0]):
    #         for y in range(result_image.shape[1]):
    #             result_image[x, y] = 0
        
    #     for p in convex_hull_points:
    #         x = p[0][0]
    #         y = p[0][1]
    #         w = 10
    #         for u in range(x - w, x + w):
    #             for v in range(y - w, y + w):
    #                 if (u >= result_image.shape[0] or v >= result_image.shape[1] or u < 0 or v < 0):
    #                     continue
    #                 result_image[u,v] = 255      
        
    return points

def thresh_callback(src, threshold):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3,3))
    height = src.shape[0]
    width = src.shape[1]
    max_thresh = 255
    # cv2.createTrackbar('Canny thresh:', "source_window", threshold, max_thresh, thresh_callback)

    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(canny_output, cv2.MORPH_CLOSE, kernel)
    
    _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = filter(lambda cont: cv2.arcLength(cont, False) > 10, contours)
    num_cnt = 0
    for i, c in enumerate(contours):
        if ( cv2.arcLength(c, False) < 500):
            continue
        num_cnt += 1
    
    contours_poly = [None]*num_cnt
    boundRect = [None]*num_cnt
    centers = [None]*num_cnt
    radius = [None]*num_cnt
    index = -1
    for i, c in enumerate(contours):
        if ( cv2.arcLength(c, False) < 500):
            continue
        index += 1
        contours_poly[index] = cv2.approxPolyDP(c, 3, True)
        boundRect[index] = cv2.boundingRect(contours_poly[index])
        centers[index], radius[index] = cv2.minEnclosingCircle(contours_poly[index])

    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    hierarchy = hierarchy[0]
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
        if ( cv2.arcLength(c, False) < 500):
            continue
        if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
            cv2.drawContours(drawing, contours, i, color[counter%5], 2)
        else:
            cv2.drawContours(drawing, contours, i, (0, 255, 0), 2)
            
        y, x, h, w, = boundRect[counter]
        if (w < 20 or h < 20):
            continue
        cropped_objects.append(src[(int)(get_min(x,0)):(int)(get_max(x+w, height)), (int)(get_min(y, 0)):(int)(get_max(y+h, width))])
        # cropped_objects.append(get_rect([1,1]))
        counter += 1
    
    print("Длинное Количество контуров", counter)
    
    # x,y,w,h 
    
    return drawing, cropped_objects

def get_min(x, bound):
    if (x > bound + 10):
        return x-10
    return bound

def get_max(x, bound):
    if (x < bound - 10):
        return x+10
    return bound

def read_image(filename, directory):
    path = os.path.join(directory, filename)
    original_image = imread(path)
    return original_image

def cut_edges(original_image, cut_threshold):
    h, w, _ = original_image.shape
    cropped_image = original_image[(int)(h*cut_threshold):(int)(h*(1-cut_threshold)), (int)(w*cut_threshold):(int)(w*(1-cut_threshold))]
    return cropped_image

def match_points_via_orb_detector(src, dst, good_match_threshold, verbose=True):   
    orb_detector = cv2.ORB_create(nfeatures=1000000)
    kp1, des1 = orb_detector.detectAndCompute(src, None)
    kp2, des2 = orb_detector.detectAndCompute(dst, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    matches = sorted(matches, key=lambda x:x.distance)

    good_matches = []
    for m in matches:
        #print(m.distance)
        if m.distance < good_match_threshold:
            good_matches.append(m)
    
    # print(len(good_matches))
    # if len(good_matches) > 4:
    #     img_matches = cv2.drawMatches(src, kp1, dst, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(img_matches, interpolation='nearest')
    #     plt.show()
    
    return good_matches     


def find_objects_on_img(path, verbose=False):
    objects_convex_hull = []
    # TODO get hull
    return objects_convex_hull



if __name__ == "__main__":
    projectDirectory = "C:\\Users\\Nikita\\Intelligent-placer-and-checker"
    placerDirectory = "PlacerDataset"
    fullDirectory = join(projectDirectory, placerDirectory)
    polygon_points = [[0, 0], [0, 3800], [1200, 3800], [1500, 1500], [1500, 1000], [1300, 0]]

    allFiles = [f for f in listdir(fullDirectory) if isfile(join(fullDirectory, f))]
    print(len(allFiles))

    thresh = 100
    cropped_objects = []
    for i in range(len(allFiles)):
    
        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        
        filename = allFiles[i]
        image = read_image(filename, fullDirectory)
        #image = crop_by_mask(image)
    
        result, cropped_objects_i = thresh_callback(image, thresh)

        ax.flatten()[0].imshow(result)
        ax.flatten()[1].imshow(image)    
        
        fig1, ax1 = plt.subplots(1, len(cropped_objects_i), figsize=(15, 10))
        for i, image in enumerate(cropped_objects_i):
            ax1.flatten()[i].imshow(image)
        cropped_objects.append(cropped_objects_i)
    
    single_item_dataset_directory = join(projectDirectory,"ML2021Dataset")
    single_item_files = [f for f in listdir(single_item_dataset_directory) if isfile(join(single_item_dataset_directory, f))]
    placer_dataset_directory = fullDirectory
    placer_files = [f for f in listdir(placer_dataset_directory) if isfile(join(placer_dataset_directory, f))]

    for batch in cropped_objects:
        best_matches = []
        best_matches_len = []
        best_matches_imgs = []
        best_matches_polygons = []
        for fragment in batch:
            src = cv2.cvtColor(fragment, cv2.COLOR_BGR2GRAY)
            best_match = 0
            best_match_len = 0
            best_img = cv2.imread(join(single_item_dataset_directory, single_item_files[0]))
            best_i = 1
            for i, single_item_file in enumerate(single_item_files, 0):
                dst = cv2.imread(join(single_item_dataset_directory, single_item_file))
                # dst = cut_edges(dst, 0.13)
                dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
                matches = match_points_via_orb_detector(src, dst, 35)
                len_matches = len(matches)
                if len_matches > best_match_len:
                    best_match_len = len_matches
                    best_img = dst
                    best_match = matches
                    best_i = i
            best_matches.append(best_match)
            best_matches_len.append(best_match_len)
            best_matches_imgs.append(best_img)
            best_matches_polygons.append(predefined_convex_hulls[best_i])
        # for l, img, src in zip(best_matches_len, best_matches_imgs, batch):
        #     print(l)
        #     fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        #     ax.flatten()[0].imshow(src)
        #     ax.flatten()[1].imshow(img)    
        #     plt.show()
        placer(polygon_points,best_matches_polygons)

    # polygon_points = [[0, 0], [0, 1900], [600, 1900], [750, 750], [750, 500], [650, 0]]
    # placer(polygon_points,[],1000,300,1)