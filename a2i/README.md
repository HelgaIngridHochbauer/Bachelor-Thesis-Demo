# ArchaeoAstroInsight Plugin - Multi-Point Clustering Feature

## Overview

This QGIS plugin enables archaeoastronomy studies by computing azimuth and horizon altitude for bearing lines. The plugin now supports **multi-point clustering**, allowing you to capture multiple objects, group them by proximity, and calculate orientations once per cluster rather than per object.

## Features

- **Single Object Mode**: Original behavior - click twice to define an object and get immediate results
- **Batch Clustering Mode**: Capture multiple objects, group them into clusters, and calculate once per cluster centroid
- **Visual Clustering**: See cluster boundaries with green circles and centroid markers
- **Efficient Calculations**: Reduce computation time by calculating only for cluster centroids
- **Comprehensive CSV Export**: Save all cluster and object data to a single CSV file

## How the Clustering System Works

### Step 1: Object Capture (Batch Mode)

When batch mode is enabled and you click twice on the map:

1. **First click**: Stores point 1
2. **Second click**: Stores point 2
3. The system:
   - Draws a line between the two points
   - Calculates the azimuth for that object (for reference only)
   - Stores both points as one object: `(point1, point2)`
   - Adds red markers at both points
   - Resets to capture the next object

**Important**: No full calculations are performed yet - objects are only stored.

### Step 2: Clustering (When You Click "Run Clustering")

1. **Extract Object Centroids**:
   - For each captured object, calculate the midpoint (centroid) of its two points
   - Example: Object with points at (lat1, lon1) and (lat2, lon2) → Centroid at ((lat1+lat2)/2, (lon1+lon2)/2)

2. **Group by Proximity**:
   - Uses DBSCAN clustering algorithm (or optimized fallback) with a 100-meter threshold
   - Objects within ~100 meters are grouped into the same cluster
   - Each cluster gets a unique ID

3. **Calculate Cluster Centroids**:
   - For each cluster, calculate the mean of all object centroids
   - This becomes the cluster centroid location

4. **Visual Representation**:
   - Green circle marker at each cluster centroid
   - Green circle polygon covering all objects in the cluster

### Step 3: Calculations (Only for Cluster Centroids)

For each cluster, **ONE calculation** is performed at the cluster centroid:

1. **Average Azimuth**:
   - Calculate azimuth for each object in the cluster
   - Take the mean of those azimuths
   - This represents the cluster's average orientation

2. **Horizon Altitude**:
   - Use the cluster centroid location (lat, lon)
   - Request horizon data from HeyWhatsThat.com API
   - Run `script.py` with the centroid location and mean azimuth
   - Get the horizon altitude at that azimuth

3. **Declination**:
   - Use the existing `computeDeclination()` function
   - Inputs: altitude, mean azimuth, and centroid location
   - Output: celestial declination

4. **Celestial Bodies**:
   - Check which stars/celestial bodies match the declination
   - Uses the same BSC5 star catalog and sun/moon checks as the original plugin

### Step 4: Results and Saving

1. **Display**:
   - Results grouped by Cluster ID
   - One result per cluster (not per object)
   - Shows: cluster ID, location, number of objects, azimuth, altitude, declination, celestial bodies

2. **CSV Export**:
   - One file with all results
   - **Cluster rows**: Full calculations (azimuth, altitude, declination, stars)
   - **Object rows**: Reference data (location, azimuth only, no altitude/declination)

## Usage Instructions

### Enabling Batch Mode

1. Click the **"A2i toggle batch mode"** toolbar button
2. Message bar will show: "Batch mode enabled. Click twice per object to capture."
3. Console will print: "Batch mode enabled"

### Capturing Objects

1. With batch mode enabled, click twice on the map to define each object
2. After the second click:
   - A line will be drawn between the two points
   - Red cross markers will appear at both points
   - Console will show: "Object captured. Total objects: X"
3. Repeat to capture multiple objects
4. Objects should be within ~100 meters of each other to be clustered together

### Running Clustering

1. After capturing all objects, click the **"A2i run clustering"** button
2. The system will:
   - Group objects into clusters (within 100m threshold)
   - Create visual representations (green circles and markers)
   - Calculate results for each cluster centroid
   - Display results in console and message bar

### Viewing Results

- **Console**: Detailed results for each cluster
- **Message Bar**: Summary message
- **Map**: Green circles show cluster boundaries, green markers show centroids

### Saving Results

1. If RESULTS_PATH is configured, a save dialog will appear automatically
2. Choose a location and filename for the CSV file
3. The file will contain:
   - One row per cluster with full calculations
   - One row per object with basic information (for reference)

### Clearing Points

Click the **"A2i clear points"** button to:
- Clear all captured objects
- Remove all markers and cluster circles
- Reset the system

## Clustering Parameters

- **Distance Threshold**: 100 meters (~0.0009 degrees)
  - Objects within this distance are grouped into the same cluster
  - Based on the centroids (midpoints) of the objects
  - Can be adjusted in the code if needed

- **Algorithm**: 
  - Primary: DBSCAN (if sklearn is available)
  - Fallback: Optimized union-find based clustering

## Key Points

✅ **Calculations are performed ONCE per cluster** - at the cluster centroid location  
✅ **Individual objects are NOT calculated** - they're only used to:
   - Determine cluster membership
   - Calculate average azimuth

✅ **The astronomical math remains unchanged** - uses the same functions as the original plugin  
✅ **The cluster centroid represents the mean location** of all objects in that cluster

## Example Scenario

If you capture 5 objects that form 2 clusters:

- **Cluster 1**: 3 objects → 1 calculation at Cluster 1 centroid
- **Cluster 2**: 2 objects → 1 calculation at Cluster 2 centroid
- **Total**: 2 calculations (instead of 5)

This significantly reduces computation while using the cluster's representative location and average orientation.

## Visualization

- **Red Cross Markers**: Individual object points
- **Green Circle Markers**: Cluster centroids
- **Green Circle Polygons**: Cluster boundaries (showing which objects belong together)
- **Lines**: Connections between object points

## Requirements

- QGIS 3.x
- Internet connection (for HeyWhatsThat.com API calls)
- Python with required packages
- Optional: sklearn for DBSCAN clustering (falls back to optimized method if not available)

## Troubleshooting

### Objects Not Clustering

- Objects may be too far apart (>100 meters)
- Check console output for cluster count
- Try capturing objects closer together

### Freezing Issues

- The plugin now processes events during long operations
- If freezing occurs, try with fewer objects first
- Check console for error messages

### No Results

- Ensure batch mode is enabled
- Capture at least one object (two clicks)
- Check internet connection for API calls
- Verify Python path is configured correctly

## Technical Details

### Coordinate Systems

- **Map Canvas**: EPSG:3857 (Web Mercator)
- **Calculations**: EPSG:4326 (WGS84 Lat/Lon)
- Automatic coordinate transformation between systems

### Performance Optimizations

- Union-Find data structure for efficient clustering
- Early termination in distance calculations
- Cached coordinate transforms
- Event processing to prevent UI freezing
- Reduced buffer segments for faster rendering

## Author Notes

The clustering feature maintains the original plugin's mathematical accuracy while providing efficient batch processing capabilities for field work with multiple related objects.

