# Clustering Methods

This folder contains clustering algorithms for grouping archaeological objects by proximity.

## Structure

- `base_clustering.py`: Base class that all clustering methods must inherit from
- `dbscan_clustering.py`: DBSCAN clustering (requires sklearn)
- `simple_clustering.py`: Simple distance-based clustering (no dependencies)
- `method1.py`, `method2.py`, `method3.py`: Template files for new clustering methods

## Adding a New Clustering Method

1. Create a new file (e.g., `kmeans_clustering.py`) in this folder
2. Copy the template from `method1.py` as a starting point
3. Implement the `cluster()` method following the interface:
   - **Input**: `object_centroids` (list of tuples: `(lat, lon, obj_points, obj_transformed)`)
   - **Parameters**: `distance_threshold` (float, in meters)
   - **Output**: `labels` (list of integers, one per object, indicating cluster membership)
4. Update `__init__.py` to register your new method:
   ```python
   from .kmeans_clustering import KMeansClustering
   
   AVAILABLE_METHODS = {
       'dbscan': DBSCANClustering,
       'simple': SimpleClustering,
       'kmeans': KMeansClustering,  # Add your method here
   }
   ```

## Interface Specification

All clustering methods must:

1. Inherit from `BaseClusteringMethod`
2. Implement `__init__()` that sets:
   - `self.name`: Human-readable name (e.g., "K-Means")
   - `self.description`: Brief description
3. Implement `cluster(object_centroids, distance_threshold=100.0)` that returns a list of cluster labels

### Input Format

`object_centroids` is a list where each element is a tuple:
- `lat` (float): Latitude of object centroid
- `lon` (float): Longitude of object centroid  
- `obj_points`: Original map points (QgsPointXY objects)
- `obj_transformed`: Transformed points (QgsPointXY objects in EPSG:4326)

### Output Format

Return a list of integers, one per object, indicating cluster membership:
- Objects in the same cluster have the same label
- Labels should start from 0 and be consecutive
- Example: `[0, 0, 0, 1, 1]` means objects 0,1,2 are in cluster 0, and objects 3,4 are in cluster 1

## Available Methods

- **DBSCAN**: Density-Based Spatial Clustering (requires sklearn)
- **Simple**: Distance-based clustering with union-find (no dependencies)

## Usage in QGIS

1. Click "A2i select clustering method" button to choose an algorithm
2. The selected method will be used when you click "A2i run clustering"

