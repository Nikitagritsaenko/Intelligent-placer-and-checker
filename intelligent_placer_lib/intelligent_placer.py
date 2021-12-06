from intelligent_placer_lib import placer, objects_carver


def check_image(path, polygon=None, verbose=False):
    if polygon is None:
        raise Exception("polygon is empty")

    objects_names, objects_convex_hull = objects_carver.find_objects_on_img(path)

    print(objects_names)
    result = placer.can_pack(polygon, objects_names, objects_convex_hull, verbose=verbose)
    return result
