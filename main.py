import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import spatial

from algorithm import ACO_TSP


# вычисление длины пути
def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


def main(size_pop, number_iteration):
    # создание объекта алгоритма муравьиной колонии
    aca = ACO_TSP(func=cal_total_distance, n_dim=num_points,
                  size_pop=size_pop,  # количество муравьёв
                  max_iter=number_iteration, distance_matrix=distance_matrix)
    best_x, best_y = aca.run()

    # Вывод результатов на экран
    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    for index in range(0, len(best_points_)):
        ax[0].annotate(best_points_[index], (best_points_coordinate[index, 0], best_points_coordinate[index, 1]))
    ax[0].plot(best_points_coordinate[:, 0],
               best_points_coordinate[:, 1], 'o-r')
    pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    # изменение размера графиков
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.show()


if __name__ == "__main__":
    num_points = 100  # количество вершин
    size_pop = 100  # количество муравьев
    number_iteration = 10  # количество итераций
    points_coordinate = np.random.rand(num_points, 2)  # генерация рандомных вершин
    print("Координаты вершин:\n", points_coordinate[:10], "\n")

    # вычисление матрицы расстояний между вершин
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    print("Матрица расстояний:\n", distance_matrix)

    start_time = time.time()  # сохранение времени начала выполнения
    main(size_pop, number_iteration)  # выполнение кода
    print("time of execution: %s seconds" % abs(time.time() - start_time))  # вычисление времени выполнения
