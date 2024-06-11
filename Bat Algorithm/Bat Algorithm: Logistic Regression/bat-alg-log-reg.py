import numpy as np
import random
import csv
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.special import expit  # logistic function (sigmoid)
from sklearn.preprocessing import StandardScaler

def adjust_parameters(dimensions):
    num_bats = 30 + 10 * dimensions
    max_iterations = 100 + 20 * dimensions
    return num_bats, max_iterations

frequency_min = 0.0
frequency_max = 2.5
alpha = 0.95  
gamma = 0.95  
initial_loudness = 1.0
initial_pulse_rate = 0.5
mutation_rate = 0.2

def fitness(data, weights, dimensions, class_labels):
    X = np.array([[row[dim] for dim in dimensions] for row in data])
    y = np.array([class_labels.index(row['class']) for row in data])
    z = np.dot(X, weights)
    y_pred = expit(z)  # apply log func
    y_pred_labels = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y, y_pred_labels)
    return -accuracy  # minimize neg accuracy to maximize accuracy

def BatAlgorithm(data, dimensions, search_space, mutation_rate, class_labels):
    num_bats, max_iterations = adjust_parameters(len(dimensions))
    
    positions = np.random.uniform(low=search_space[0], high=search_space[1], size=(num_bats, len(dimensions), len(class_labels)))
    velocities = np.zeros_like(positions)
    frequencies = np.zeros(num_bats)
    loudness = np.full(num_bats, initial_loudness)
    pulse_rate = np.full(num_bats, initial_pulse_rate)
    
    gbest_position = positions[np.argmin([fitness(data, pos, dimensions, class_labels) for pos in positions])]
    gbest_fitness = fitness(data, gbest_position, dimensions, class_labels)
    
    for iteration in range(max_iterations):
        for i in range(num_bats):
            frequencies[i] = frequency_min + (frequency_max - frequency_min) * random.random()
            velocities[i] += (positions[i] - gbest_position) * frequencies[i]
            positions[i] += velocities[i]
            
            if random.random() > pulse_rate[i]:
                positions[i] = gbest_position + 0.05 * np.random.randn(len(dimensions), len(class_labels))
            
            positions[i] = np.clip(positions[i], search_space[0], search_space[1])
            reinit_probability = 0.1 * (1 - iteration / max_iterations)
            if random.random() < reinit_probability:
                positions[i] = np.random.uniform(low=search_space[0], high=search_space[1], size=(len(dimensions), len(class_labels)))
            
            if random.random() < mutation_rate:
                positions[i] += np.random.uniform(-5, 5, size=(len(dimensions), len(class_labels)))
                positions[i] = np.clip(positions[i], search_space[0], search_space[1])

            current_fitness = fitness(data, positions[i], dimensions, class_labels)
            if current_fitness < gbest_fitness and loudness[i] > random.random():
                gbest_position = positions[i].copy()
                gbest_fitness = current_fitness
                loudness[i] *= alpha
                pulse_rate[i] *= (1 - np.exp(-gamma * iteration))
    return gbest_position

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
                    row_data['class'] = value
            data.append(row_data)
        
        # Standard scaling
        scaler = StandardScaler()
        feature_keys = [key for key in data[0].keys() if key != 'class']
        feature_matrix = np.array([[row[key] for key in feature_keys] for row in data])
        scaled_features = scaler.fit_transform(feature_matrix)
        
        for i, row in enumerate(data):
            for j, key in enumerate(feature_keys):
                row[key] = scaled_features[i, j]
        return data

# main
start_time = time.time()
dataset = read_dataset('/Users/angelesmarin/Desktop/ba & spark/ba datasets/IRIS.csv')

# find numerical columns as features; non-numerical as class labels
dimension_keys = [key for key in dataset[0].keys() if key != 'class']
search_space = (-1, 1)  # Adjusting search space for weights
labels = sorted(set(row['class'] for row in dataset))

# split dataset into training & validation sets
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# optimizing weights using ba
optimal_weights = BatAlgorithm(train_data, dimension_keys, search_space, mutation_rate, labels)

print("Optimal Weights:")
print(optimal_weights)

# pred labels using optimal weights
def predict_labels(data, weights, dimensions, class_labels):
    X = np.array([[row[dim] for dim in dimensions] for row in data])
    z = np.dot(X, weights)
    y_pred = expit(z)  # apply log function
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_pred_labels = np.array([class_labels[label] if label < len(class_labels) else None for label in y_pred_labels])
    return y_pred_labels

# generate true labels & pred labels for validation set
y_val_true = np.array([row['class'] for row in val_data])
y_val_pred = predict_labels(val_data, optimal_weights, dimension_keys, labels)

# get validation metrics
val_accuracy = accuracy_score(y_val_true, y_val_pred)
val_precision = precision_score(y_val_true, y_val_pred, average='macro', zero_division=0)
val_recall = recall_score(y_val_true, y_val_pred, average='macro', zero_division=0)
val_f1 = f1_score(y_val_true, y_val_pred, average='macro', zero_division=0)

print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Precision: {val_precision * 100:.2f}%")
print(f"Validation Recall: {val_recall * 100:.2f}%")
print(f"Validation F1 Score: {val_f1 * 100:.2f}%")

# generate true labels & pred labels for training set
y_train_true = np.array([row['class'] for row in train_data])
y_train_pred = predict_labels(train_data, optimal_weights, dimension_keys, labels)

# get test accuracy
test_accuracy = accuracy_score(y_train_true, y_train_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# sample predictions
sample_indices = np.random.choice(len(val_data), 3, replace=False)
for i in sample_indices:
    sample = val_data[i]
    sample_features = [sample[dim] for dim in dimension_keys]
    sample_true_class = sample['class']
    sample_pred_class = predict_labels([sample], optimal_weights, dimension_keys, labels)[0]
    print(f"Sample {i + 1}: Features = {sample_features}, Predicted Class = {sample_pred_class}, True Class = {sample_true_class}")

print(f"Time taken (seconds): {time.time() - start_time}")
