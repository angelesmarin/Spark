import csv
import numpy as np
import time

def read_dataset(filename):
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        data = []
        for row in reader:
            row_data = {}
            for key, value in row.items():
                try:
                    row_data[key] = float(value)
                except ValueError:
                    row_data[key] = value
            data.append(row_data)
        return data

def determine_label_column(data):
    for key in data[0].keys():
        if not isinstance(data[0][key], float):
            return key
    return None

def calculate_centroids(data, dimensions, label_column):
    label_to_points = {}
    
    for row in data:
        label = row[label_column]
        if label not in label_to_points:
            label_to_points[label] = []
        label_to_points[label].append([row[dim] for dim in dimensions])
    
    centroids = {}
    for label, points in label_to_points.items():
        centroids[label] = np.mean(points, axis=0)
    
    return centroids

if __name__ == "__main__":
    start_time = time.time()
    dataset = read_dataset('/Users/angelesmarin/Desktop/IRIS.csv')  

    # dynamically determine label column
    label_column = determine_label_column(dataset)
    if label_column is None:
        print("No label column found.")
    else:
        dimension_keys = [key for key in dataset[0].keys() if key != label_column]
        centroids = calculate_centroids(dataset, dimension_keys, label_column)

        print("Final Centroids:")
        for label, centroid in centroids.items():
            print(f"Label {label}:")
            for feature, value in zip(dimension_keys, centroid):
                print(f"  {feature}: {value}")

        print(f"Time taken (seconds): {time.time() - start_time}")
