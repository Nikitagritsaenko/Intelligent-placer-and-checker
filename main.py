from intelligent_placer_lib import intelligent_placer


def run_intelligent_placer(path, polygon):
    return intelligent_placer.check_image(path, polygon, verbose=True)


if __name__ == '__main__':
    polygon_points = [[40, 0], [30, 90], [90, 125], [155, 140], [120, -40]]
    for i in range(1, 6):
        path = "PlacerDataset/" + str(i) + ".jpg"
        print(run_intelligent_placer(path, polygon_points))
