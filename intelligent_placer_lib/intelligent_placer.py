from intelligent_placer_lib import placer, objects_carver


def check_image(path, polygon=None, verbose=False):
    __, objects_convex_hull = objects_carver.find_objects_on_img(path)
    if polygon is None:
        raise Exception("polygon is empty")

    result = placer.can_pack(polygon, objects_convex_hull, verbose=verbose)
    return result
