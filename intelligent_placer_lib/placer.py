import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

task_counter = 0


class MyPolygon:
    def __init__(self, points, name):
        self.name = name
        self.normalized_points = MyPolygon.normalize_to_origin(points)
        self.points = self.normalized_points.copy()
        self.center = [0, 0]
        self.alpha = 0
        self.sh_polygon = Polygon(self.normalized_points)
        self.area = self.sh_polygon.area

    def is_inside(self, point):
        p1 = Point(point[0], point[1])
        return self.sh_polygon.contains(p1)

    def is_figure_inside(self, points):
        return Polygon(points).within(self.sh_polygon)

    def is_overlap(self, another):
        return self.sh_polygon.intersects(another.sh_polygon)

    def rotate_points(self, alpha):
        c_x, c_y = self.center[0], self.center[1]
        for i in range(len(self.points)):
            point = self.points[i]
            x, y = point[0] - c_x, point[1] - c_y
            cos = np.cos(alpha)
            sin = np.sin(alpha)
            self.points[i] = [c_x + x * cos - y * sin, c_y + x * sin + y * cos]

    def rotate(self, alpha):
        self.rotate_points(alpha)
        self.alpha = self.alpha + alpha
        self.sh_polygon = Polygon(self.points)

    def set_center(self, new_center):
        for i in range(len(self.normalized_points)):
            p = self.normalized_points[i]
            self.points[i] = [p[0] + new_center[0], p[1] + new_center[1]]
        self.center = new_center
        self.rotate_points(self.alpha)
        self.sh_polygon = Polygon(self.points)

    @staticmethod
    def normalize_to_origin(points):
        center = MyPolygon.get_center(points)
        normalized_points = []
        for p in points:
            normalized_points.append([p[0] - center[0], p[1] - center[1]])
        return normalized_points

    @staticmethod
    def get_center(points):
        x_arr = [item[0] for item in points]
        y_arr = [item[1] for item in points]
        return [(min(x_arr) + max(x_arr)) / 2, (min(y_arr) + max(y_arr)) / 2]

    def draw(self, ax):
        x_arr = [item[0] for item in self.points]
        y_arr = [item[1] for item in self.points]
        ax.plot(x_arr + [x_arr[0]], y_arr + [y_arr[0]])

    def find_intersections(self, ready_objects):
        intersections = []
        for ready_obj in ready_objects:
            if self is ready_obj:
                continue
            if self.is_overlap(ready_obj):
                intersections.append(ready_obj)
        return intersections


def plot_configuration(polygon, objects, filename=""):
    fig, ax = plt.subplots()
    ax.plot()

    x_arr = [item[0] for item in polygon]
    y_arr = [item[1] for item in polygon]
    ax.plot(x_arr + [x_arr[0]], y_arr + [y_arr[0]])

    for i in range(len(objects)):
        obj = objects[i]
        obj.draw(ax)

    plt.gca().set_aspect("equal")
    if filename != "":
        plt.savefig(filename)
    else:
        plt.savefig("solution_" + str(task_counter))
    plt.close(fig)
    # fig.show()


def find_borders(polygon):
    x_arr = [item[0] for item in polygon.points]
    y_arr = [item[1] for item in polygon.points]
    return [min(x_arr), max(x_arr), min(y_arr), max(y_arr)]


def drop_point(borders):
    x1, x2, y1, y2 = borders[0], borders[1], borders[2], borders[3]
    x = np.random.randint(x1, x2)
    y = np.random.randint(y1, y2)
    return [x, y]


def try_to_reconfigure(polygon, ready_objects, obj_to_move, new_object, K=50):
    old_center = obj_to_move.center
    move_radius = np.sqrt(obj_to_move.area / np.pi)
    all_objects = []
    for o in ready_objects:
        all_objects.append(o)
    all_objects.append(new_object)

    for i in range(K):
        x = np.random.randint(old_center[0] - move_radius / 2, old_center[0] + move_radius / 2)
        y = np.random.randint(old_center[1] - move_radius / 2, old_center[1] + move_radius / 2)
        obj_to_move.set_center([x, y])
        if not polygon.is_figure_inside(obj_to_move.points):
            continue
        if len(obj_to_move.find_intersections(all_objects)) == 0:
            print("получилось сдвинуть объект " + obj_to_move.name +
                  ", который пересекался с новым объектом " + new_object.name)
            return True
    obj_to_move.set_center(old_center)
    return False


def try_to_put_object(obj, polygon, ready_objects, M):
    borders = find_borders(polygon)

    alpha_step = 30
    for alpha in range(0, 180, alpha_step):

        # пытаемся M раз положить предмет
        for j in range(M):
            # бросаем случайную точку
            center_point = drop_point(borders)

            # проверяем, что точка внутри многоугольника, если нет, то continue
            if not polygon.is_inside(center_point):
                continue
            obj.set_center(center_point)
            obj.rotate(alpha_step)

            # проверяем, что фигура находится внутри м.у
            if not polygon.is_figure_inside(obj.points):
                continue
            # проверяем, что предмет не будет пересекаться с уже уложенными
            intersections = obj.find_intersections(ready_objects)

            if j > M / 2 and len(intersections) == 1:
                if try_to_reconfigure(polygon, ready_objects, intersections[0], obj):
                    assert len(obj.find_intersections(ready_objects)) == 0
                    ready_objects.append(obj)
                    print([r_o.name for r_o in ready_objects])
                    return True

            if len(intersections) == 0:
                ready_objects.append(obj)
                return True
    return False


def placer(polygon_points, objects_names, objects, N=500, M=100, verbose=False):
    global task_counter
    task_counter = task_counter + 1

    figures = []
    for i in range(len(objects)):
        points = objects[i]
        name = objects_names[i]
        figures.append(MyPolygon(points, name))
    polygon = MyPolygon(polygon_points, "многоугольник")

    area_sum = sum([figure.area for figure in figures])
    if area_sum > polygon.area:
        return False

    for i in range(N):
        if i % 20 == 0:
            if verbose:
                print(i)

        figures = []
        for i in range(len(objects)):
            points = objects[i]
            name = objects_names[i]
            figures.append(MyPolygon(points, name))
        figures.sort(key=lambda f: f.area, reverse=True)

        # объекты, которые уже уложили
        ready_objects = []

        # пытаемся уложить все объекты
        for obj in figures:
            if not try_to_put_object(obj, polygon, ready_objects, M):
                break

        if len(ready_objects) == len(figures):
            if verbose:
                plot_configuration(polygon.points, ready_objects)
            return True
    if verbose:
        plot_configuration(polygon.points, ready_objects)
    return False


def can_pack(polygon_points, objects_names, objects, verbose=False):
    return placer(polygon_points, objects_names, objects, verbose=verbose)


def get_rect(rectangle_params, center_point=[0, 0]):
    w, h = rectangle_params[0], rectangle_params[1]
    c_x, c_y = center_point[0], center_point[1]
    p1, p2, p3, p4 = [c_x + w / 2, c_y + h / 2], [c_x - w / 2, c_y + h / 2], [c_x - w / 2, c_y - h / 2], [c_x + w / 2,
                                                                                                          c_y - h / 2]
    return p1, p2, p3, p4


def get_ideal_polygon(radius, vertex_num):
    x = [radius * np.cos(2 * np.pi / vertex_num * i) for i in range(vertex_num)]
    y = [radius * np.sin(2 * np.pi / vertex_num * i) for i in range(vertex_num)]

    result = [[x[i], y[i]] for i in range(len(x))]
    return result


if __name__ == '__main__':
    result = can_pack([[0, 0], [0, 2200], [600, 2200], [750, 750], [750, 500], [650, 0]], [

        get_rect([50, 50]),
        get_ideal_polygon(100, 6),
        get_ideal_polygon(100, 3),
        get_ideal_polygon(200, 4),

        [[0, 0], [0, 200], [200, 200], [200, 0], [100, -100]],
        [[0, 0], [0, 200], [100, 300], [200, 200], [200, 0], [100, -100]],
        get_rect([400, 400]),
        get_rect([425, 400]),
        get_rect([380, 400]),

    ])
