from faker import Faker
import random
import csv

fake = Faker()

def generate_dataset(num_points):
    dataset = []
    for _ in range(num_points):
        label = random.choice(['A', 'B', 'C'])
        if label == 'A':
            feature1 = random.uniform(10, 40)
            feature2 = random.uniform(10, 40)
            feature3 = random.uniform(10, 40)
            feature4 = random.uniform(10, 40)
            feature5 = random.uniform(10, 40)
        elif label == 'B':
            feature1 = random.uniform(60, 90)
            feature2 = random.uniform(60, 90)
            feature3 = random.uniform(60, 90)
            feature4 = random.uniform(60, 90)
            feature5 = random.uniform(60, 90)
        else:  # label == 'C'
            feature1 = random.uniform(120, 150)
            feature2 = random.uniform(120, 150)
            feature3 = random.uniform(120, 150)
            feature4 = random.uniform(120, 150)
            feature5 = random.uniform(120, 150)
        
        dataset.append((feature1, feature2, feature3, feature4, feature5, label))
    return dataset

def save_dataset_to_csv(dataset, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'class'])
        for data in dataset:
            writer.writerow(data)

if __name__ == "__main__":
    num_points = 1000  # You can adjust the number of points as needed
    dataset = generate_dataset(num_points)
    desktop_path = '/Users/angelesmarin/Desktop/2C_5D_dataset.csv'
    save_dataset_to_csv(dataset, desktop_path)
    print(f"Dataset with {num_points} points generated and saved to '{desktop_path}'")
