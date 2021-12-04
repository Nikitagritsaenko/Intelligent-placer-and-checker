from intelligent_placer_lib import intelligent_placer


def run_intelligent_placer(path, polygon):
    return intelligent_placer.check_image(path, polygon, verbose=True)


if __name__ == '__main__':
    path = "PlacerDataset/4.jpg"
    polygon_points = [[0, 0], [0, 3800], [2200, 3800], [2600, 2600], [2600, 2100], [2400, 0]]
    print(run_intelligent_placer(path, polygon_points))
