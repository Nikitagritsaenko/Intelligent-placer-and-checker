import placer
import objects_carver


def check_image(path, polygon=None):
    objects_convex_hull = objects_carver.find_objects_on_img(path)
    if polygon is None:
        raise Exception("polygon is empty")

    result = placer.can_pack(polygon, objects_convex_hull)
    return result


if __name__ == '__main__':
    path = "C:\\Users\\Nikita\\Intelligent-placer-and-checker"
    ans = check_image(path, [[1, 1], [1, 1], [1, 1]])

    print(ans)
