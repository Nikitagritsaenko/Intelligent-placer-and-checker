from intelligent_placer_lib import intelligent_placer
import os
import matplotlib.pyplot as plt
import numpy as np

def run_intelligent_placer(path, polygon):
    return intelligent_placer.check_image(path, polygon, verbose=False)


def get_all_dataset_filenames(dir_path):
    files = os.listdir(dir_path)
    image_names = filter(lambda x: x.endswith('.jpg'), files)
    result = [os.path.join(dir_path, name) for name in image_names]
    return result

save_dir = "PlacerOutput"

def draw_polygon(polygon, filename=""):
    fig, ax = plt.subplots()
    ax.plot()

    x_arr = [item[0] for item in polygon]
    y_arr = [item[1] for item in polygon]
    ax.plot(x_arr + [x_arr[0]], y_arr + [y_arr[0]])

    try:
        if filename != "":
            plt.savefig(filename)
        else:
            plt.savefig(save_dir + "\polygon_")
    except FileNotFoundError as e:
        print(f"can`t save fig: {e}'")
    fig.show()

def get_all_polygon_filenames(dir_path):

    files = os.listdir(dir_path)
    image_names = filter(lambda x: x.startswith('polygon'), files)
    result = [os.path.join(dir_path, name) for name in image_names]
    return result

import json


def read_polygon_from_file(dir_path):
    result = []
    with open(dir_path, "r") as fd:
        for line in fd.readlines():
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace(",", "")
            nums_str = line.strip().split(" ")
            nums = filter(lambda x: x != "", nums_str)
            arr = [float(value) for value in nums]
            result.append(arr)
    return result

polygons_dict = {
    "3": [[40, 0],
          [30, 90],
          [90, 125],
          [155, 140],
          [120, -40]],

    "4": [[21.8, 106.4],
          [30.4, 68.6],
          [73.2, 10.6],
          [180.0, 62.0],
          [124.6, 166.4]],

    "5": [[20.71, 101.08],
          [28.88, 65.16999999999999],
          [69.54, 10.069999999999999],
          [171.0, 58.9],
          [118.36999999999999, 158.07999999999998]],

    "6": [[0.0, 0.0],
          [125, 0],
          [125, 100],
          [0, 100]],

    "7": [[0.0, 0.0],
          [135, 0],
          [135, 110],
          [0, 110]],

    "8": [[0.0, 0.0],
          [135, 0],
          [115, 110],
          [0, 110]],

    "9": [[0.0, 0.0],
          [205, 0],
          [205, 65],
          [0, 65]],

    "10": [[0, 0],
           [0, 173],
           [55, 173],
           [130, 100],
           [107, 0]],

    "11": [[103, 0],
           [0, 75],
           [120, 175],
           [223, 75]],

    "12": [[103, 0],
           [0, 75],
           [120, 175],
           [223, 75]],

    "13": [[0.0, 0.0],
           [205, 0],
           [205, 62],
           [0, 62]],
}

# фотография 2 точилка определяется как карта
# фотография 3 - всё норм

test_dict_True = [
    {"img": "PlacerDataset/1.jpg",
     "poly": polygons_dict["4"]},
    {"img": "PlacerDataset/2.jpg",
     "poly": polygons_dict["3"]},
    {"img": "PlacerDataset/3.jpg",
     "poly": polygons_dict["3"]},
    {"img": "PlacerDataset/1.jpg",
     "poly": polygons_dict["6"]},
    {"img": "PlacerDataset/1.jpg",
     "poly": polygons_dict["7"]},
    {"img": "PlacerDataset/1.jpg",
     "poly": polygons_dict["8"]},
    {"img": "PlacerDataset/1.jpg",
     "poly": polygons_dict["9"]},
    {"img": "PlacerDataset/3.jpg",
     "poly": polygons_dict["10"]},
    {"img": "PlacerDataset/2.jpg",
     "poly": polygons_dict["10"]},
    {"img": "PlacerDataset/3.jpg",
     "poly": polygons_dict["11"]},
    {"img": "PlacerDataset/2.jpg",
     "poly": polygons_dict["12"]},

]

test_dict_False = [
    {"img": "PlacerDataset/1.jpg",
     "poly": polygons_dict["5"]},
    {"img": "PlacerDataset/1.jpg",
     "poly": polygons_dict["13"]},

]

def test():

    def draw_poly(poly):
        x_arr = [item[0] for item in poly]
        y_arr = [item[1] for item in poly]
        ax.plot(x_arr + [x_arr[0]], y_arr + [y_arr[0]])


    true_ans_count = 0
    false_ans_count = 0

    for test_param in test_dict_True:
        path = test_param["img"]
        polygon_points = test_param["poly"]

        print(test_param)
        fig, ax = plt.subplots()
        draw_poly(polygon_points)
        # draw_poly(polygon_points2)
        fig.show()
        ans = run_intelligent_placer(path, polygon_points)

        print(f"True: {ans}")
        if ans:
            true_ans_count += 1
        else:
            false_ans_count += 1

    print("true_ans_count", true_ans_count)
    print("false_ans_count", false_ans_count)




def main():


    test()
    return
    files = get_all_dataset_filenames("PlacerDataset/CheckerImages")
    print(files)
    polygons = [read_polygon_from_file(filename) for filename in get_all_polygon_filenames("PlacerDataset")]
    [draw_polygon(poly) for poly in polygons]

    for path in files:
        # for polygon_points in polygons[0]:
        polygon_points = polygons[0]
        print(run_intelligent_placer(path, polygon_points))
    return

    polygon_points = [[40, 0], [30, 90], [90, 125], [155, 140], [120, -40]]
    for i in range(1, 2):
        path = "PlacerDataset/" + str(i) + ".jpg"
        print(run_intelligent_placer(path, polygon_points))




if __name__ == '__main__':
    main()

