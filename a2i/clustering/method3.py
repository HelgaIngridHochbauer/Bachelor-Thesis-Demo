"""
Clustering Method 3 Template

TODO: Implement your clustering algorithm here.
"""

from .base_clustering import BaseClusteringMethod


class Method3Clustering(BaseClusteringMethod):
    """Method 3: [Describe your method here]"""
    
    def __init__(self):
        super().__init__()
        self.name = "Method 3"
        self.description = "[Describe your clustering method]"
    
    def cluster(self, object_centroids, distance_threshold=100.0):
        """
        Cluster objects using Method 3.
        
        Args:
            object_centroids: List of tuples, each containing:
                - lat (float): Latitude of object centroid
                - lon (float): Longitude of object centroid
                - obj_points: Original map points (QgsPointXY objects)
                - obj_transformed: Transformed points (QgsPointXY objects in EPSG:4326)
            distance_threshold: Distance threshold in meters (default: 100m)
        
        Returns:
            labels: List of integers, one per object, indicating cluster membership.
                   Objects in the same cluster have the same label.
                   Labels should start from 0 and be consecutive.
        
        Example:
            If you have 5 objects that form 2 clusters:
            - Objects 0, 1, 2 are in cluster 0
            - Objects 3, 4 are in cluster 1
            Then return: [0, 0, 0, 1, 1]
        """
        n = len(object_centroids)
        if n == 0:
            return []
        if n == 1:
            return [0]
        
        # Convert meters to approximate degrees
        # 1 degree ≈ 111,000 meters at equator
        eps_degrees = distance_threshold / 111000.0
        
        print(f"Method 3: Clustering {n} objects with threshold: {eps_degrees:.6f} degrees (~{distance_threshold} meters)")
        
        # TODO: Implement your clustering algorithm here
        # For now, return a placeholder (each object in its own cluster)
        labels = list(range(n))
        
        unique_labels = len(set(labels))
        print(f"Method 3 found {unique_labels} cluster(s)")
        
        return labels

