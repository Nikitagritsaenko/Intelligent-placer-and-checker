import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import threading
import time


class MyPolygon:
    def __init__(self, points):
        self.normalized_points = MyPolygon.normalize_to_origin(points)
        self.points = self.normalized_points.copy()
        self.center = [0, 0]
        self.sh_polygon = Polygon(self.normalized_points)
        self.square = self.sh_polygon.area

    def is_inside(self, point):
        p1 = Point(point[0], point[1])
        return self.sh_polygon.contains(p1)

    def is_figure_inside(self, points):
        return Polygon(points).within(self.sh_polygon)

    def is_overlap(self, another):
        return self.sh_polygon.intersects(another.sh_polygon)

    def rotate(self, alpha):
        c_x, c_y = self.center[0], self.center[1]
        for i in range(len(self.points)):
            point = self.points[i]
            x, y = point[0] - c_x, point[1] - c_y
            cos = np.cos(alpha)
            sin = np.sin(alpha)
            self.points[i] = [c_x + x * cos - y * sin, c_y + x * sin + y * cos]
        self.sh_polygon = Polygon(self.points)

    def set_center(self, new_center):
        for i in range(len(self.points)):
            p = self.normalized_points[i]
            self.points[i] = [p[0] + new_center[0], p[1] + new_center[1]]
        self.center = new_center
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

    def draw_fig(self):
        fig, ax = plt.subplots()
        ax.plot()
        x_arr = [item[0] for item in self.points]
        y_arr = [item[1] for item in self.points]
        ax.plot(x_arr + [x_arr[0]], y_arr + [y_arr[0]])

        fig.show()

    def is_intersects(self, ready_objects):
        for ready_obj in ready_objects:
            if self.is_overlap(ready_obj):
                return True
        return False


def plot_configuration(polygon, objects, title=None, filename=None):
    fig, ax = plt.subplots()
    ax.plot()

    x_arr = [item[0] for item in polygon]
    y_arr = [item[1] for item in polygon]
    ax.plot(x_arr + [x_arr[0]], y_arr + [y_arr[0]])

    for i in range(len(objects)):
        obj = objects[i]
        obj.draw(ax)

    plt.gca().set_aspect("equal")
    if title is not None:
        ax.set_title(title)

    fig.show()

    if filename is not None:
        fig.savefig(filename)



def find_borders(polygon):
    x_arr = [item[0] for item in polygon.points]
    y_arr = [item[1] for item in polygon.points]
    return [min(x_arr), max(x_arr), min(y_arr), max(y_arr)]


def drop_point(borders):
    x1, x2, y1, y2 = borders[0], borders[1], borders[2], borders[3]

    x = np.random.randint(x1, x2)
    y = np.random.randint(y1, y2)
    return [x, y]


def try_to_put_object(obj, polygon, ready_objects, M):
    borders = find_borders(polygon)

    alpha_step = 15

    for alpha in range(-90, 90, alpha_step):
        # пытаемся M раз положить предмет

        for j in range(M):
            if glob_ready_flag:
                return
            # бросаем случайную точку
            center_point = drop_point(borders)
            # проверяем, что точка не попала внутрь уложенных п.у
            # for rect in ready_objects:
            #    if is_inside_polygon(rect, center_point):
            #       continue
            # проверяем, что точка внутри многоугольника, если нет, то continue

            if not polygon.is_inside(center_point):
                continue

            obj.set_center(center_point)
            obj.rotate(alpha)

            # проверяем, что фигура находится внутри м.у
            if not polygon.is_figure_inside(obj.points):
                continue
            # проверяем, что п.у не будет пересекаться с уже уложенными
            if obj.is_intersects(ready_objects):
                continue
            ready_objects.append(obj)

            return True
    return False


glob_ready_objects = []
glob_ready_flag = False


def one_iter(polygon, figures, M):
    self_id = threading.currentThread().ident
    # объекты, которые уже уложили
    ready_objects = []

    # пытаемся уложить все объекты
    for obj in figures:
        if not try_to_put_object(obj, polygon, ready_objects, M):
            print(f'{self_id})\t can`t put object\n', end="")
            # plot_configuration(polygon.points, ready_objects)
            break
        # plot_configuration(polygon.points, ready_objects)
        # TODO 4: попробовать двигать предметы - алгоритм тетриса

    return ready_objects


def thread_task(polygon_points, objects, N, M):
    global glob_ready_objects
    global glob_ready_flag

    figures = [MyPolygon(points) for points in objects]
    polygon = MyPolygon(polygon_points)
    figures = sorted(figures, key=lambda x: x.square, reverse=True)
    self_id = threading.currentThread().ident
    for i in range(N):
        if i % 1 == 0:
            print(f"{self_id})\t {i}\n", end="")
        ready_objects = one_iter(polygon, figures, M)

        if len(ready_objects) == len(figures):
            plot_configuration(polygon.points, ready_objects, title="result configuration", filename=f"good_config_N={self_id}.png")
            return

    # plot_configuration(polygon.points, ready_objects)
    return




def placer(polygon_points, objects, N=1000, M=300, W=50):
    figures = [MyPolygon(points) for points in objects]
    polygon = MyPolygon(polygon_points)

    area_sum = sum([figure.square for figure in figures])
    if area_sum > polygon.square:
        print("ВСЁ ПЛОХО")
        return "no"

    plot_configuration(polygon.points, figures)
    # TODO: сортируем предметы по площади

    figures = sorted(figures, key=lambda x: x.square, reverse=True)


    workers = [threading.Thread(target=thread_task, args=(polygon_points, objects, int(N / W), M)) for i in range(W)]

    timer_start = time.time()
    for i in range(W):
        workers[i].start()

    for i in range(W):
        workers[i].join()

    timer_end = time.time()
    elapsed = timer_end - timer_start
    print(f"elapsed: {elapsed}")

    if glob_ready_flag:
        plot_configuration(polygon.points, glob_ready_objects, title="result configuration", filename=f"good_config_N.png")
        return "yes"


    plot_configuration(polygon.points, glob_ready_objects)
    return "no"


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
    result = placer([[0, 0], [0, 1900], [600, 1900], [750, 750], [750, 500], [650, 0]], [

        get_rect([50, 50]),
        get_ideal_polygon(100, 6),
        get_ideal_polygon(100, 3),
        get_ideal_polygon(200, 4),

        [[0, 0], [0, 200], [200, 200], [200, 0], [100, -100]],
        [[0, 0], [0, 200], [100, 300], [200, 200], [200, 0], [100, -100]],
        get_rect([400, 400]),
        get_rect([425, 400]),
        get_rect([380, 400]),


        ],10000,300,1)
    '''
    get_ideal_polygon(100, 6),
    get_ideal_polygon(100, 3),
    get_ideal_polygon(100, 5),
    get_ideal_polygon(100, 30),
    get_ideal_polygon(100, 30),
    get_ideal_polygon(100, 3),
    get_ideal_polygon(200, 30),
    get_ideal_polygon(50, 5),
    '''
    print(result)