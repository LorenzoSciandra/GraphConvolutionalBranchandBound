import random2
import time
import numpy as np
from py2opt.solver import Solver

class RouteFinder:
    def __init__(self, distance_matrix, cities_names, iterations=5, writer_flag=False, method='py2opt', return_to_begin=False, verbose=True):
        self.distance_matrix = distance_matrix
        self.iterations = iterations
        self.return_to_begin = return_to_begin
        self.writer_flag = writer_flag
        self.cities_names = cities_names
        self.verbose = verbose
        self.bound = np.sqrt(len(cities_names)) * 0.00127

    def solve(self):
        start_time = round(time.time() * 1000)
        elapsed_time = 0
        iteration = 0
        best_distance = 0
        best_route = []

        while iteration < self.iterations:
            num_cities = len(self.distance_matrix)
            if self.verbose:
                print(round(elapsed_time), 'msec')
            initial_route = [0] + random2.sample(range(1, num_cities), num_cities - 1)
            if self.return_to_begin:
                initial_route.append(0)
            tsp = Solver(self.distance_matrix, initial_route)
            new_route, new_distance, distances = tsp.two_opt(improvement_threshold=self.bound)

            if iteration == 0:
                best_distance = new_distance
                best_route = new_route
            else:
                pass

            if new_distance < best_distance:
                best_distance = new_distance
                best_route = new_route

            elapsed_time = round(time.time() * 1000) - start_time
            iteration += 1

        if self.writer_flag:
            self.writer(best_route, best_distance, self.cities_names)

        if self.cities_names:
            best_route = [self.cities_names[i] for i in best_route]
            return best_distance, best_route
        else:
            return best_distance, best_route

    def solve_from_init_cycle(self, start_cycle):
        tsp = Solver(self.distance_matrix, start_cycle)
        route, distance, _ = tsp.two_opt(improvement_threshold=self.bound)
        return distance, route

    @staticmethod
    def writer(best_route, best_distance, cities_names):
        f = open("../results.txt", "w+")
        for i in best_route:
            f.write(cities_names[i])
            f.write("\n")
            print(cities_names[i])
        f.write(str(best_distance))
        f.close()
