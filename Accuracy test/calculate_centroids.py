import csv
import numpy as np
import time

def read_dataset(filename):
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        data = []
        for row in reader:
            row_data = {key: float(value) if key != 'Class' else value for key, value in row.items()}
            data.append(row_data)
        return data

def calculate_centroids(data, dimensions):
    label_to_points = {}
    
    for row in data:
        label = row['Class']
        if label not in label_to_points:
            label_to_points[label] = []
        label_to_points[label].append([row[dim] for dim in dimensions])
    
    centroids = {}
    for label, points in label_to_points.items():
        centroids[label] = np.mean(points, axis=0)
    
    return centroids

if __name__ == "__main__":
    start_time = time.time()
    dataset = read_dataset('/Users/angelesmarin/Desktop/2Class_5D_dataset.csv')

    dimension_keys = [key for key in dataset[0].keys() if key != 'Class']
    centroids = calculate_centroids(dataset, dimension_keys)

    print("Final Centroids:")
    for label, centroid in centroids.items():
        print(f"Label {label}: {centroid}")

    print(f"Time taken (seconds): {time.time() - start_time}")
