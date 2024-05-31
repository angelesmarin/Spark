import numpy as np
import random
import csv
import time

num_bats = 100
max_iterations = 200
frequency_min = 0.0
frequency_max = 2.5
alpha = 0.97
gamma = 0.9
initial_loudness = 1.0
initial_pulse_rate = 0.5

def fitness(data, position, label, dimensions):
    distances = 0
    for row in data:
        if row['class'] == label:
            distances += np.sqrt(sum((row[dim] - position[d]) ** 2 for d, dim in enumerate(dimensions)))
    return distances

def BatAlgorithm(data, label, dimensions, search_space):
    positions = np.random.uniform(low=search_space[0], high=search_space[1], size=(num_bats, len(dimensions)))
    velocities = np.zeros_like(positions)
    frequencies = np.zeros(num_bats)
    loudness = np.full(num_bats, initial_loudness)
    pulse_rate = np.full(num_bats, initial_pulse_rate)

    gbest_position = positions[np.argmin([fitness(data, pos, label, dimensions) for pos in positions])]
    gbest_fitness = fitness(data, gbest_position, label, dimensions)
    tolerance = 0.001

    for iteration in range(max_iterations):
        for i in range(num_bats):
            frequencies[i] = frequency_min + (frequency_max - frequency_min) * random.random()
            velocities[i] += (positions[i] - gbest_position) * frequencies[i]
            positions[i] += velocities[i]

            if random.random() > pulse_rate[i]:
                positions[i] = gbest_position + 0.001 * np.random.randn(len(dimensions))

            positions[i] = np.clip(positions[i], search_space[0], search_space[1])

            current_fitness = fitness(data, positions[i], label, dimensions)
            if current_fitness < gbest_fitness and loudness[i] > random.random():
                gbest_position = positions[i].copy()
                gbest_fitness = current_fitness
                loudness[i] *= alpha
                pulse_rate[i] *= (1 - np.exp(-gamma * iteration))

        if gbest_fitness < tolerance:
            break

    return gbest_position  

def read_dataset(filename):
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        data = []
        for row in reader:
            row_data = {key: float(value) if key != 'class' else value for key, value in row.items()}
            data.append(row_data)
        return data

def compute_centroids(data, label, dimensions, search_space):
    centroid = BatAlgorithm(data, label, dimensions, search_space)
    return centroid

start_time = time.time()
dataset = read_dataset('/Users/angelesmarin/Desktop/2Class_5Feature_dataset.csv')

dimension_keys = [key for key in dataset[0].keys() if key != 'class']
search_space = (0, 100)  # Set search space boundaries for all dimensions

labels = set(row['class'] for row in dataset)
centroids = {label: compute_centroids(dataset, label, dimension_keys, search_space) for label in labels}

print("Final Centroids:")
for label, centroid in centroids.items():
    print(f"Label {label}: {centroid}")

print(f"Time taken (seconds): {time.time() - start_time}")
