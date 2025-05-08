import random as rd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def auto_silhouette(data, max_clusters=10):
    best_num_clusters = 2
    best_silhouette_score = -1
    silhouette_scores = []

    for num_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        current_score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(current_score)

        if current_score > best_silhouette_score:
            best_silhouette_score = current_score
            best_num_clusters = num_clusters

    return best_num_clusters


def calculate_euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))


class MyKMeans:
    def __init__(self, data: Optional[np.ndarray] = None, n_clusters: int = 2):
        self.data = np.array(data) if data is not None and not isinstance(data, np.ndarray) else data
        self.n_clusters = n_clusters
        self.centroids = []
        self.cluster_labels = []

    def initialize_centroids(self) -> List[np.ndarray]:
        centroids = []

        initial_index = rd.randint(0, len(self.data) - 1)
        centroids.append(self.data[initial_index])


        for _ in range(1, self.n_clusters):
            distances = []
            for data_point in self.data:
                min_distance = min(calculate_euclidean_distance(data_point, centroid) for centroid in centroids)
                distances.append(min_distance)

            total_distance = sum(distances)
            probabilities = [d / total_distance for d in distances]

            new_centroid_index = np.random.choice(len(self.data), p=probabilities)
            centroids.append(self.data[new_centroid_index])
        return centroids

    def assign_to_clusters(self) -> List[int]:
        labels = []
        for data_point in self.data:
            distances_to_centroids = [calculate_euclidean_distance(data_point, centroid) for centroid in
                                      self.centroids]
            closest_centroid_index = np.argmin(distances_to_centroids)
            labels.append(closest_centroid_index)
        return labels

    def update_centroid_positions(self) -> bool:
        new_centroids = []
        for cluster_index in range(self.n_clusters):
            cluster_points = self.data[np.array(self.cluster_labels) == cluster_index]
            if len(cluster_points) == 0:
                new_centroids.append(self.centroids[cluster_index])
            else:
                new_centroids.append(np.mean(cluster_points, axis=0))

        if np.allclose(self.centroids, new_centroids):
            return False

        self.centroids = new_centroids
        return True

    def visualize_clusters(self, feature1_index: int = 0, feature2_index: int = 1):
        plt.scatter(self.data[:, feature1_index], self.data[:, feature2_index],
                    c=self.cluster_labels, cmap='tab10', alpha=0.7)
        plt.scatter(np.array(self.centroids)[:, feature1_index],
                    np.array(self.centroids)[:, feature2_index],
                    marker='o', s=70, c='red', edgecolors='black')
        plt.xlabel(f'feature {feature1_index}')
        plt.ylabel(f'feature {feature2_index}')
        plt.show()

    def predict_clusters(self, new_data: List[List[float]]) -> List[int]:
        new_data = np.array(new_data)
        predicted_labels = []
        for data_point in new_data:
            distances = [calculate_euclidean_distance(data_point, centroid) for centroid in self.centroids]
            predicted_labels.append(np.argmin(distances))
        return predicted_labels

    def fit_model(self, max_iterations: int = 100) -> None:
        if self.data is None:
            raise ValueError("input data is required")

        self.centroids = self.initialize_centroids()

        for iteration in range(max_iterations):
            self.cluster_labels = self.assign_to_clusters()
            self.visualize_clusters()

            if not self.update_centroid_positions():
                print(f"converged after {iteration + 1} iterations")
                break

    def visualize_all_projections(self):
        num_features = self.data.shape[1]

        plt.figure(figsize=(num_features * 3, num_features * 3))
        for y_feature in range(num_features):
            for x_feature in range(num_features):
                if x_feature == y_feature:
                    continue

                plt.subplot(num_features, num_features, y_feature * num_features + x_feature + 1)
                plt.scatter(self.data[:, x_feature], self.data[:, y_feature],
                            c=self.cluster_labels, cmap='tab10', alpha=0.7)
                plt.scatter(np.array(self.centroids)[:, x_feature],
                            np.array(self.centroids)[:, y_feature],
                            marker='o', s=70, c='red', edgecolors='black')
                plt.xlabel(f'feature {x_feature}')
                plt.ylabel(f'feature {y_feature}')

        plt.tight_layout()
        plt.show()


def main():
    iris_dataset = load_iris()
    data = iris_dataset.data

    optimal_clusters = auto_silhouette(data)

    print("optimal_clusters:", optimal_clusters)

    custom_kmeans = MyKMeans(data=data, n_clusters=optimal_clusters)
    custom_kmeans.fit_model()

    custom_kmeans.visualize_all_projections()


if __name__ == "__main__":
    main()
