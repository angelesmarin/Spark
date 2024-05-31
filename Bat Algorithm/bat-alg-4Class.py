
import numpy as np
import random
import csv
import time
import matplotlib.pyplot as plt

num_bats = 100
max_iterations = 200
frequency_min = 0.0
frequency_max = 2.5
alpha = 0.97
gamma = 0.9
initial_loudness = 1.0
initial_pulse_rate = 0.5

search_space = [(0, 100), (0, 100)]

def fitness(data, position, label):
    centroid_x, centroid_y = position
    distances = 0
    for row in data:
        if row['Class'] == label:
            distances += ((row['x'] - centroid_x) ** 2 + (row['y'] - centroid_y) ** 2) ** 0.5
    return distances

def BatAlgorithm(data, label):
    positions = np.random.uniform(low=search_space[0][0], high=search_space[0][1], size=(num_bats, 2))
    velocities = np.zeros_like(positions)
    frequencies = np.zeros(num_bats)
    loudness = np.full(num_bats, initial_loudness)
    pulse_rate = np.full(num_bats, initial_pulse_rate)

    gbest_position = positions[np.argmin([fitness(data, pos, label) for pos in positions])]
    gbest_fitness = fitness(data, gbest_position, label)
    tolerance = 0.001  

    fitness_over_time = []

    for iteration in range(max_iterations):
        for i in range(num_bats):
            frequencies[i] = frequency_min + (frequency_max - frequency_min) * random.random()
            velocities[i] += (positions[i] - gbest_position) * frequencies[i]
            positions[i] += velocities[i]

            if random.random() > pulse_rate[i]:
                positions[i] = gbest_position + 0.001 * np.random.randn(2)

            positions[i] = np.clip(positions[i], search_space[0][0], search_space[0][1])

            current_fitness = fitness(data, positions[i], label)
            if current_fitness < gbest_fitness and loudness[i] > random.random():
                gbest_position = positions[i].copy()
                gbest_fitness = current_fitness
                loudness[i] *= alpha
                pulse_rate[i] *= (1 - np.exp(-gamma * iteration))

        fitness_over_time.append(gbest_fitness)

        if gbest_fitness < tolerance:
            break

    return gbest_position, fitness_over_time  

def read_dataset(filename):
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        return [dict(row, **{'x': float(row['x']), 'y': float(row['y'])}) for row in reader]

def compute_centroids(data, label):
    centroid, fitness_over_time = BatAlgorithm(data, label)
    return centroid, fitness_over_time

start_time = time.time()
dataset = read_dataset('/Users/angelesmarin/Desktop/4Cluster2Ddataset.csv')

centroid_A, fitness_A = compute_centroids(dataset, 'A')
centroid_B, fitness_B = compute_centroids(dataset, 'B')
centroid_C, fitness_C = compute_centroids(dataset, 'C')
centroid_D, fitness_D = compute_centroids(dataset, 'D')

print("Final Centroids:")
print(f"Label A: {centroid_A}")
print(f"Label B: {centroid_B}")
print(f"Label C: {centroid_C}")
print(f"Label D: {centroid_D}")

print(f"Time taken (seconds): {time.time() - start_time}")
