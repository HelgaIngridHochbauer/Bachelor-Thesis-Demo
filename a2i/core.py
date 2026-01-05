SCRIPT_PATH = "C:\\"
#Archaeo-Astro Insight#

#################################################
# Global parameters                            #
#################################################
QGIS_CRS = "EPSG:3857" #canvas coordinates
TARGET_CRS = "EPSG:4326" #coordinates of your map

RESULTS_PATH =''
PYTHON_PATH = ''
LINE_WIDTH = 0.7

SCRIPT_SLEEP = 10

DOWNLOAD_MAP = True 
MAP_TYPE = "mt1.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}"

#################################################
#sys.path.append(SCRIPT_PATH)
from .resources import *
import requests
import time
import subprocess
import csv
import os.path
from os import path
from qgis.gui import QgsMapToolEmitPoint
from qgis.core import QgsProject
from qgis.core import QgsCoordinateReferenceSystem
from qgis.core import QgsCoordinateTransform
from qgis.core import Qgis
from qgis.gui import QgsMapTool
from qgis.core import QgsPoint
from qgis.core import QgsVectorLayer
from qgis.core import QgsFeature
from qgis.core import QgsGeometry
from qgis.core import QgsField
from qgis.gui import QgsVertexMarker
from qgis.PyQt.QtCore import Qt, QVariant, QCoreApplication

from qgis.PyQt.QtGui import QIcon, QColor
from qgis.PyQt import QtGui
from qgis.PyQt.QtWidgets import QInputDialog, QLineEdit, QPushButton, QMessageBox, QProgressDialog
from qgis.core import QgsPointXY, QgsRectangle
from .utility import *
from .dialog import *
from .save_data import *
from pathlib import Path

#Main tool
class DeclinationTool(QgsMapToolEmitPoint):
    def __init__(self, canvas, iface, plugin_dir, resultsPath, PythonPath, scriptSleep, lineWidth, downloadMap):
        self.pointList = []
        self.transformedPoints = []
        self.code = ""
        self.az = 0
        self.altitude = 0
        self.decl = 0
        self.stars = []
        self.canvas = canvas
        self.iface = iface
        self.scriptPath = plugin_dir
        # Batch mode state management
        self.batch_mode = False
        self.captured_objects = []  # List of tuples: ((point1, point2), (transformed1, transformed2))
        self.markers = []  # List of QgsVertexMarker objects for visualization
        self.cluster_results = []  # Store cluster calculation results
        self.cluster_layers = []  # List of QgsVectorLayer for cluster visualizations (not used anymore, kept for compatibility)
        
        global RESULTS_PATH 
        RESULTS_PATH = resultsPath
        global PYTHON_PATH
        PYTHON_PATH = PythonPath
        global LINE_WIDTH
        LINE_WIDTH = lineWidth
        global SCRIPT_SLEEP
        SCRIPT_SLEEP = scriptSleep
        global DOWNLOAD_MAP
        DOWNLOAD_MAP = downloadMap

        QgsMapToolEmitPoint.__init__(self, self.canvas)

    def canvasPressEvent( self, e ):
        #get point on click
        point = self.toMapCoordinates(self.canvas.mouseLastXY())

        #transform from map CRS to target CRS
        tr = QgsCoordinateTransform(QgsCoordinateReferenceSystem(QGIS_CRS), QgsCoordinateReferenceSystem(TARGET_CRS), QgsProject.instance())
        transformed_point = tr.transform(point)
        
        #append points to respective lists
        self.pointList.append(point)
        self.transformedPoints.append(transformed_point)

        if len(self.transformedPoints) == 1:
            print('Point 1: ({:.4f}, {:.4f})'.format(transformed_point[1], transformed_point[0]))
        else:
            print('Point 2: ({:.4f}, {:.4f})'.format(transformed_point[1], transformed_point[0]))

        #print(point)
        #print(transformed_point)
        if len(self.pointList) == 2:
            self.drawLine()
            self.az = computeAzimuth(self.pointList)
            
            # If batch mode is enabled, store the object and reset for next object
            if self.batch_mode:
                # Store the object (both points)
                object_points = (self.pointList[0], self.pointList[1])
                object_transformed = (self.transformedPoints[0], self.transformedPoints[1])
                self.captured_objects.append((object_points, object_transformed))
                
                # Add markers for visualization
                self.addObjectMarkers(object_points)
                
                # Reset for next object
                self.pointList = []
                self.transformedPoints = []
                print(f"Object captured. Total objects: {len(self.captured_objects)}")
    
    def canvasReleaseEvent( self, e ):
        # Only process immediately if not in batch mode
        if not self.batch_mode and len(self.pointList) == 2:
            self.handleRequest()
            self.iface.messageBar().clearWidgets()
            self.iface.messageBar().pushSuccess("Success","[HeyWhatsThat.com] response received")

            if (self.code == ""):
                return
            
            self.iface.messageBar().clearWidgets()
            self.iface.messageBar().pushMessage("Running horizon Python script, please wait....", Qgis.Info)
            self.handleScript()
            self.iface.messageBar().clearWidgets()
            self.iface.messageBar().pushSuccess("Success","Horizon Python script finished succesfuly")

            self.decl = computeDeclination(self.altitude, self.az, self.transformedPoints)

            self.stars = checkDeclinationBSC5(self.decl, self.scriptPath)
            sunMoon = checkDeclinationSunMoon(self.decl)

            if sunMoon != "None":
                self.stars.append(sunMoon)

            write_to_csv(self, self.scriptPath, self.transformedPoints[0].x(), self.transformedPoints[0].y(), self.az, self.altitude, self.decl, self.stars)

            self.pointList = []
            self.transformedPoints = []
    
    #draw the line and compute azimuth
    def drawLine(self):

        #create layer for the line
        start_point = QgsPoint(self.pointList[0].x(),self.pointList[0].y())
        end_point = QgsPoint(self.pointList[1].x(),self.pointList[1].y())
        line_layer = QgsVectorLayer('LineString?crs=epsg:3857', 'line', 'memory')
        # setAbstract is deprecated, use setComment instead if available, or skip
        try:
            line_layer.setComment('Point one ({:.4f},{:.4f}) and point two ({:.4f},{:.4f})'.format(
                self.transformedPoints[0].y(), self.transformedPoints[0].x(), self.transformedPoints[1].y(), self.transformedPoints[1].x()))
        except:
            # Fallback for older QGIS versions
            try:
                line_layer.setAbstract('Point one ({:.4f},{:.4f}) and point two ({:.4f},{:.4f})'.format(
                    self.transformedPoints[0].y(), self.transformedPoints[0].x(), self.transformedPoints[1].y(), self.transformedPoints[1].x()))
            except:
                pass  # Skip if neither method works
        line_layer.renderer().symbol().setWidth(LINE_WIDTH)
        pr = line_layer.dataProvider()
        seg = QgsFeature()
        seg.setGeometry(QgsGeometry.fromPolyline([start_point, end_point]))
        pr.addFeatures([ seg ])
        QgsProject.instance().addMapLayers([line_layer])

            
            #gLine = QgsGeometry.fromPolyline([QgsPoint(self.pointList[0].x(),self.pointList[0].y()), QgsPoint(self.pointList[1].x(),self.pointList[1].y())])
            #self.rubberBand.setToGeometry(gLine, None)
            #print(self.pointList)
    
    #send request for HeyWhatsThat.com code
    def handleRequest(self):
        self.iface.messageBar().pushMessage("Sending HTTP request to [HeyWhatsThat.com]. Please wait for the response.....", Qgis.Info, 2)
        point = QgsPoint(self.transformedPoints[0].x(),self.transformedPoints[0].y())
        #print(point)
        req = "http://heywhatsthat.com/bin/query.cgi?lat={0:.4f}&lon={1:.4f}1&name={2}".format(self.transformedPoints[0].y(), self.transformedPoints[0].x(), "Horizon1")
        print(req) 
        #req_test = "http://heywhatsthat.com/bin/query.cgi?lat=44.297147&lon=69.129591&name={}".format("Horizon1")
        r = requests.get(req)
        #print(r)
        i = 0
        while r.text == "" and i <= 10:
            time.sleep(2)
            r = requests.get(req)
            print("[HeyWhatsThat.com] Waiting for server response, please wait...")
            i += 1
        print("[HeyWhatsThat.com] Horizon profile code is " + r.text.strip("\n"))
        self.code = r.text.strip("\n")
        
     
    #call script and get altitude value   
    def handleScript(self):
        script_path = os.path.join(self.scriptPath, "script.py")
        #print(script_path)
        #print(PYTHON_PATH)
        args = [PYTHON_PATH, script_path, self.code, str(self.az)]
        result = ''
        #print(args)
        #time.sleep(SCRIPT_SLEEP)

        result = subprocess.run(args, capture_output=True, shell=True, text=True)
        i = 1
        while result.returncode != 0 and i <= 10:
            print("Horizon Python script running, please wait...")
            time.sleep(SCRIPT_SLEEP)
            result = subprocess.run(args, capture_output=True, shell=True, text=True)
            i += 1

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Unknown error"
            print(f"Error running Python script: {error_msg}")
            raise RuntimeError(f"Failed to run horizon script: {error_msg}")
        
        try:
            altitude_value = float(result.stdout.strip())
            print("Altitude is {}".format(altitude_value))
            self.altitude = altitude_value
        except ValueError as e:
            print(f"Error parsing altitude output: {result.stdout}")
            raise ValueError(f"Could not parse altitude from script output: {e}")

    def reset(self):
        self.pointList = []
        self.transformedPoints = []
        self.isEmittingPoint = False
        if hasattr(self, 'rubberBand'):
            self.rubberBand.reset(True)
        
    def deactivate(self):
        QgsMapTool.deactivate(self)
        if hasattr(self, 'deactivated'):
            self.deactivated.emit()
    
    def addObjectMarkers(self, points):
        """Add vertex markers for object points"""
        for point in points:
            marker = QgsVertexMarker(self.canvas)
            marker.setCenter(QgsPointXY(point.x(), point.y()))
            marker.setColor(QColor(255, 0, 0))  # Red for object points
            marker.setIconSize(8)
            marker.setIconType(QgsVertexMarker.ICON_CROSS)
            marker.setPenWidth(2)
            self.markers.append(marker)
    
    def addCentroidMarker(self, point, cluster_id):
        """Add vertex marker for cluster centroid (the point where calculations are performed)"""
        marker = QgsVertexMarker(self.canvas)
        marker.setCenter(QgsPointXY(point.x(), point.y()))
        marker.setColor(QColor(0, 255, 0))  # Bright green for centroids
        marker.setIconSize(15)  # Larger size for better visibility
        marker.setIconType(QgsVertexMarker.ICON_CIRCLE)
        marker.setPenWidth(4)  # Thicker outline for visibility
        self.markers.append(marker)
        print(f"Added green centroid marker for cluster {cluster_id} at ({point.x():.2f}, {point.y():.2f})")
    
    def clearMarkers(self):
        """Remove all markers from canvas"""
        for marker in self.markers:
            try:
                # QgsVertexMarker should be removed from canvas scene
                if marker and self.canvas.scene():
                    self.canvas.scene().removeItem(marker)
            except:
                pass
        self.markers = []
    
    def clearClusterLayers(self):
        """Remove all cluster visualization layers"""
        for layer in self.cluster_layers:
            try:
                QgsProject.instance().removeMapLayer(layer.id())
            except:
                pass
        self.cluster_layers = []
    
    def process_clusters(self):
        """Group captured objects into clusters and calculate centroids"""
        if len(self.captured_objects) == 0:
            self.iface.messageBar().pushWarning("Warning", "No objects captured. Please capture objects first.")
            return []
        
        self.iface.messageBar().pushMessage("Clustering objects...", Qgis.Info, duration=2)
        
        # Allow Qt to process events
        QCoreApplication.processEvents()
        
        # Extract centroids of each object (midpoint of the two points)
        object_centroids = []
        for obj_points, obj_transformed in self.captured_objects:
            # Calculate midpoint (centroid) of the object
            p1 = obj_transformed[0]
            p2 = obj_transformed[1]
            centroid_lat = (p1.y() + p2.y()) / 2.0
            centroid_lon = (p1.x() + p2.x()) / 2.0
            object_centroids.append((centroid_lat, centroid_lon, obj_points, obj_transformed))
        
        # Allow Qt to process events
        QCoreApplication.processEvents()
        
        # Try to use sklearn DBSCAN, fallback to simple distance-based clustering
        # Threshold: 100 meters ≈ 0.0009 degrees (much larger for better clustering)
        eps_degrees = 0.0009  # ~100 meters
        
        try:
            from sklearn.cluster import DBSCAN
            import numpy as np
            
            # Prepare data for DBSCAN (lat, lon)
            coords = np.array([(c[0], c[1]) for c in object_centroids])
            
            print(f"Clustering {len(object_centroids)} objects with threshold: {eps_degrees} degrees (~100 meters)")
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=eps_degrees, min_samples=1).fit(coords)
            labels = clustering.labels_
            
            # Debug: count clusters
            unique_labels = len(set(labels))
            print(f"DBSCAN found {unique_labels} cluster(s)")
            
        except ImportError:
            # Fallback: optimized simple distance-based clustering
            print(f"Using fallback clustering for {len(object_centroids)} objects with threshold: {eps_degrees} degrees (~100 meters)")
            labels = self._simple_clustering(object_centroids, distance_threshold=100.0)
            
            # Debug: count clusters
            unique_labels = len(set(labels))
            print(f"Fallback clustering found {unique_labels} cluster(s)")
        
        # Allow Qt to process events after clustering
        QCoreApplication.processEvents()
        
        # Group objects by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(object_centroids[idx])
        
        # Cache coordinate transform (create once, reuse many times)
        tr = QgsCoordinateTransform(QgsCoordinateReferenceSystem(TARGET_CRS), 
                                   QgsCoordinateReferenceSystem(QGIS_CRS), 
                                   QgsProject.instance())
        
        # Calculate cluster centroids and prepare results
        cluster_results = []
        total_clusters = len(clusters)
        
        for cluster_idx, (cluster_id, objects) in enumerate(clusters.items()):
            # Allow Qt to process events to prevent freezing
            QCoreApplication.processEvents()
            
            # Calculate mean centroid
            mean_lat = sum(obj[0] for obj in objects) / len(objects)
            mean_lon = sum(obj[1] for obj in objects) / len(objects)
            
            # Transform back to map coordinates for visualization
            try:
                centroid_map_point = tr.transform(QgsPointXY(mean_lon, mean_lat), 
                                                 QgsCoordinateTransform.ReverseTransform)
            except Exception as e:
                print(f"Error transforming coordinates for cluster {cluster_id}: {e}")
                continue
            
            # Add centroid marker (this is the point where calculations are performed)
            self.addCentroidMarker(centroid_map_point, cluster_id)
            
            cluster_results.append({
                'cluster_id': cluster_id,
                'centroid_lat': mean_lat,
                'centroid_lon': mean_lon,
                'centroid_map_point': centroid_map_point,
                'num_objects': len(objects),
                'objects': objects
            })
        
        print(f"Found {len(cluster_results)} clusters from {len(self.captured_objects)} objects")
        return cluster_results
    
    def _simple_clustering(self, object_centroids, distance_threshold=100.0):
        """Simple distance-based clustering using union-find (fallback when sklearn not available)
        
        Optimized with union-find data structure for O(n²) worst case but much faster average case.
        
        Args:
            object_centroids: List of (lat, lon, ...) tuples
            distance_threshold: Distance threshold in meters (default 100m)
        
        Returns:
            List of cluster labels
        """
        # Convert meters to approximate degrees
        # 1 degree ≈ 111,000 meters at equator
        # distance_threshold meters ≈ (distance_threshold / 111000) degrees
        eps_degrees = distance_threshold / 111000.0  # Convert meters to degrees
        eps_squared = eps_degrees * eps_degrees  # Use squared distance to avoid sqrt
        
        n = len(object_centroids)
        if n == 0:
            return []
        if n == 1:
            return [0]
        
        # Union-Find data structure for efficient clustering
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Only check pairs that might be close (early termination optimization)
        # Pre-sort by latitude for potential spatial optimization
        sorted_indices = sorted(range(n), key=lambda i: object_centroids[i][0])
        
        # Compare points - use squared distance to avoid sqrt
        for i_idx, i in enumerate(sorted_indices):
            obj1 = object_centroids[i]
            lat1, lon1 = obj1[0], obj1[1]
            
            # Only check nearby points (within latitude range)
            for j_idx in range(i_idx + 1, n):
                j = sorted_indices[j_idx]
                obj2 = object_centroids[j]
                lat2, lon2 = obj2[0], obj2[1]
                
                # Early termination: if latitude difference is too large, skip rest
                lat_diff = lat2 - lat1
                if lat_diff > eps_degrees:
                    break  # No more points can be within threshold
                
                # Calculate squared distance (avoid sqrt for performance)
                lon_diff = lon2 - lon1
                dist_squared = lat_diff * lat_diff + lon_diff * lon_diff
                
                if dist_squared <= eps_squared:
                    union(i, j)
        
        # Assign cluster labels
        cluster_map = {}
        labels = []
        current_label = 0
        
        for i in range(n):
            root = find(i)
            if root not in cluster_map:
                cluster_map[root] = current_label
                current_label += 1
            labels.append(cluster_map[root])
        
        return labels
    
    def calculate_cluster_orientation(self, cluster_info, progress=None):
        """Calculate orientation and declination for a cluster centroid"""
        centroid_lat = cluster_info['centroid_lat']
        centroid_lon = cluster_info['centroid_lon']
        objects = cluster_info['objects']
        
        # Calculate average azimuth from all objects in cluster
        azimuths = []
        for obj_points, obj_transformed in [(o[2], o[3]) for o in objects]:
            # Calculate azimuth for this object
            az = computeAzimuth([obj_points[0], obj_points[1]])
            azimuths.append(az)
        
        # Use mean azimuth (or could use median for robustness)
        mean_az = sum(azimuths) / len(azimuths) if azimuths else 0
        
        # Process events before network request
        QCoreApplication.processEvents()
        if progress:
            progress.setLabelText(f"Requesting horizon data for cluster {cluster_info['cluster_id']}...")
            QCoreApplication.processEvents()
        
        # Request horizon code with shorter timeout and frequent event processing
        req = "http://heywhatsthat.com/bin/query.cgi?lat={0:.4f}&lon={1:.4f}&name={2}".format(
            centroid_lat, centroid_lon, f"Cluster_{cluster_info['cluster_id']}")
        
        # Use shorter timeout and process events more frequently
        r = None
        max_retries = 10
        for i in range(max_retries + 1):
            # Process events multiple times before request to keep UI responsive
            for _ in range(3):
                QCoreApplication.processEvents()
            if progress and progress.wasCanceled():
                return None
            
            try:
                # Shorter timeout (3 seconds instead of 10) to reduce blocking time
                r = requests.get(req, timeout=3)
                
                # Process events immediately after request
                for _ in range(3):
                    QCoreApplication.processEvents()
                
                if r and r.text and r.text.strip():
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Network error for cluster {cluster_info['cluster_id']}: {e}")
                for _ in range(3):
                    QCoreApplication.processEvents()
                if i >= max_retries:
                    return None
            
            if i < max_retries:
                # Wait with very frequent event processing (every 0.1 second)
                for wait_step in range(10):  # Break 1 second into 10 steps
                    QCoreApplication.processEvents()
                    if progress and progress.wasCanceled():
                        return None
                    time.sleep(0.1)
        
        if not r or not r.text or not r.text.strip():
            return None
        
        code = r.text.strip("\n")
        if code == "":
            return None
        
        # Process events before subprocess
        QCoreApplication.processEvents()
        if progress:
            progress.setLabelText(f"Calculating altitude for cluster {cluster_info['cluster_id']}...")
            QCoreApplication.processEvents()
        
        # Get altitude from script with shorter timeout
        script_path = os.path.join(self.scriptPath, "script.py")
        args = [PYTHON_PATH, script_path, code, str(mean_az)]
        
        # Use subprocess with shorter timeout and better event handling
        result = None
        max_retries = 5  # Reduced retries
        for i in range(max_retries + 1):
            try:
                QCoreApplication.processEvents()
                if progress and progress.wasCanceled():
                    return None
                
                # Shorter timeout (30 seconds instead of 60)
                result = subprocess.run(args, capture_output=True, shell=True, text=True, timeout=30)
                
                QCoreApplication.processEvents()
                
                if result.returncode == 0:
                    break
                    
            except (subprocess.TimeoutExpired, TypeError) as e:
                print(f"Subprocess timeout/error for cluster {cluster_info['cluster_id']}: {e}")
                QCoreApplication.processEvents()
                if i >= max_retries:
                    return None
            
            if i < max_retries and result and result.returncode != 0:
                # Wait with frequent event processing
                wait_time = min(SCRIPT_SLEEP, 3)  # Cap at 3 seconds
                wait_steps = int(wait_time * 5)  # Break into 0.2 second steps
                for wait_step in range(wait_steps):
                    QCoreApplication.processEvents()
                    if progress and progress.wasCanceled():
                        return None
                    time.sleep(0.2)
        
        if not result or result.returncode != 0:
            return None
        
        try:
            altitude = float(result.stdout.strip())
        except ValueError:
            return None
        
        # Calculate declination
        # computeDeclination expects points where y() is latitude (in EPSG:4326)
        # Create QgsPointXY with (lon, lat) so y() returns latitude
        decl_point = QgsPointXY(centroid_lon, centroid_lat)
        decl = computeDeclination(altitude, mean_az, [decl_point])
        
        # Check stars
        stars = checkDeclinationBSC5(decl, self.scriptPath)
        sunMoon = checkDeclinationSunMoon(decl)
        if sunMoon != "None":
            stars.append(sunMoon)
        
        return {
            'cluster_id': cluster_info['cluster_id'],
            'centroid_lat': centroid_lat,
            'centroid_lon': centroid_lon,
            'azimuth': mean_az,
            'altitude': altitude,
            'declination': decl,
            'stars': stars,
            'num_objects': cluster_info['num_objects'],
            'objects': objects  # Include objects for CSV export
        }
    
    def process_all_clusters(self):
        """Process all clusters and calculate results"""
        if len(self.captured_objects) == 0:
            self.iface.messageBar().pushWarning("Warning", "No objects captured. Please capture objects first.")
            return
        
        try:
            self.iface.messageBar().pushMessage("Processing clusters...", Qgis.Info, duration=3)
            
            # Create progress dialog to keep UI responsive
            progress = QProgressDialog("Processing clusters...", "Cancel", 0, 100, self.canvas)
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(False)
            progress.setAutoReset(False)
            progress.setMinimumDuration(0)  # Show immediately
            progress.show()
            QCoreApplication.processEvents()
            
            # Get clusters
            progress.setLabelText("Grouping objects into clusters...")
            progress.setValue(5)
            QCoreApplication.processEvents()
            
            clusters = self.process_clusters()
            
            if len(clusters) == 0:
                progress.close()
                self.iface.messageBar().pushWarning("Warning", "No clusters found.")
                return
            
            # Calculate results for each cluster
            self.cluster_results = []
            total_clusters = len(clusters)
            progress.setMaximum(100)
            
            for idx, cluster_info in enumerate(clusters):
                # Check if cancelled
                if progress.wasCanceled():
                    progress.close()
                    self.iface.messageBar().pushWarning("Cancelled", "Processing was cancelled by user.")
                    return
                
                # Update progress (5% for clustering, 95% for calculations, distributed across clusters)
                progress_value = 5 + int(95 * (idx + 1) / total_clusters)
                progress.setValue(progress_value)
                progress.setLabelText(f"Processing cluster {idx + 1}/{total_clusters}...")
                QCoreApplication.processEvents()  # Critical: process events before blocking call
                
                result = self.calculate_cluster_orientation(cluster_info, progress)
                if result:
                    self.cluster_results.append(result)
                
                # Process events after each cluster
                QCoreApplication.processEvents()
            
            progress.setValue(100)
            progress.setLabelText("Creating visualizations...")
            QCoreApplication.processEvents()
            
            # Display results
            self.display_cluster_results()
            
            # Centroids are already marked during process_clusters(), so no need to create polygons
            # Just refresh the canvas to show the centroid markers
            if self.canvas:
                self.canvas.refresh()
            
            progress.close()
            
        except KeyError as e:
            error_msg = f"Missing data key: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.iface.messageBar().pushCritical("Clustering Error", error_msg)
        except Exception as e:
            error_msg = f"Error during clustering: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.iface.messageBar().pushCritical("Clustering Error", error_msg)
    
    def display_cluster_results(self):
        """Display cluster results grouped by Cluster ID"""
        if not self.cluster_results:
            return
        
        # Create results message
        results_text = f"=== Cluster Results ({len(self.cluster_results)} clusters) ===\n\n"
        for result in self.cluster_results:
            stars_str = ', '.join(result['stars']) if result['stars'] else 'None'
            results_text += f"Cluster {result['cluster_id']}:\n"
            results_text += f"  Location: ({result['centroid_lat']:.4f}, {result['centroid_lon']:.4f})\n"
            results_text += f"  Objects: {result['num_objects']}\n"
            results_text += f"  Azimuth: {result['azimuth']:.2f}°\n"
            results_text += f"  Altitude: {result['altitude']:.2f}°\n"
            results_text += f"  Declination: {result['declination']:.2f}°\n"
            results_text += f"  Points to: {stars_str}\n\n"
        
        print(results_text)
        
        # Show detailed message in message bar
        if len(self.cluster_results) == 1:
            result = self.cluster_results[0]
            stars_str = ', '.join(result['stars']) if result['stars'] else 'None'
            msg = f"Cluster {result['cluster_id']}: Az={result['azimuth']:.1f}° Alt={result['altitude']:.1f}° Decl={result['declination']:.1f}° → {stars_str}"
        else:
            msg = f"Processed {len(self.cluster_results)} clusters. See console for full details."
        
        self.iface.messageBar().pushSuccess("Clustering Complete", msg)
        
        # Optionally save results
        if RESULTS_PATH != "Empty":
            self.save_cluster_results()
    
    def save_cluster_results(self):
        """Save all cluster results to a single CSV file"""
        if not self.cluster_results:
            return
        
        global RESULTS_PATH
        if RESULTS_PATH == "Empty":
            RESULTS_PATH = os.getcwd()
        
        # Create a custom save dialog for batch results
        from qgis.PyQt.QtWidgets import QFileDialog
        from qgis.PyQt import QtGui
        
        logo_icon_path = ':/plugins/a2i/logo/icons/logo.png'
        filepath, _ = QFileDialog.getSaveFileName(
            None,
            "Save Cluster Results",
            RESULTS_PATH,
            "Comma Separated Values Files (*.csv)"
        )
        
        if not filepath:
            return  # User cancelled
        
        # Prepare all data for CSV
        all_rows = []
        
        # Header row
        all_rows.append(['cluster_id', 'type', 'latitude', 'longitude', 'azimuth', 'altitude', 
                        'declination', 'stars', 'num_objects', 'comments'])
        
        # Add cluster results (one row per cluster)
        for result in self.cluster_results:
            stars_str = ', '.join(result['stars']) if result['stars'] else 'None'
            all_rows.append([
                result['cluster_id'],
                'cluster_centroid',
                result['centroid_lat'],
                result['centroid_lon'],
                result['azimuth'],
                result['altitude'],
                result['declination'],
                stars_str,
                result['num_objects'],
                f'Cluster {result["cluster_id"]} with {result["num_objects"]} objects'
            ])
        
        # Add individual object information (for reference)
        for cluster_result in self.cluster_results:
            cluster_id = cluster_result['cluster_id']
            # Check if 'objects' key exists (it should after the fix above)
            if 'objects' not in cluster_result:
                print(f"Warning: No objects data for cluster {cluster_id}, skipping object rows")
                continue
            
            objects = cluster_result['objects']
            
            for obj_idx, obj in enumerate(objects):
                try:
                    obj_points, obj_transformed = obj[2], obj[3]
                    # Get midpoint of object
                    obj_lat = (obj_transformed[0].y() + obj_transformed[1].y()) / 2.0
                    obj_lon = (obj_transformed[0].x() + obj_transformed[1].x()) / 2.0
                    
                    # Calculate azimuth for this object
                    az = computeAzimuth([obj_points[0], obj_points[1]])
                    
                    all_rows.append([
                        cluster_id,
                        'object',
                        obj_lat,
                        obj_lon,
                        az,
                        '',  # altitude not calculated for individual objects
                        '',  # declination not calculated for individual objects
                        '',  # stars not calculated for individual objects
                        '',  # num_objects not applicable
                        f'Object {obj_idx + 1} in cluster {cluster_id}'
                    ])
                except Exception as e:
                    print(f"Error processing object {obj_idx} in cluster {cluster_id}: {e}")
                    continue
        
        # Write to CSV
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                data_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerows(all_rows)
            
            num_clusters = len(self.cluster_results)
            num_objects = len(self.captured_objects)
            self.iface.messageBar().pushSuccess("Saved", 
                f"Saved {num_clusters} clusters and {num_objects} objects to {os.path.basename(filepath)}")
            print(f"Saved {num_clusters} clusters and {num_objects} objects to {filepath}")
        except IOError as e:
            error_msg = f"File I/O error: {str(e)}"
            self.iface.messageBar().pushWarning("Save Error", error_msg)
            print(f"Error saving results: {e}")
        except Exception as e:
            error_msg = f"Failed to save results: {str(e)}"
            self.iface.messageBar().pushWarning("Save Error", error_msg)
            print(f"Error saving results: {e}")
    
    
    def clear_captured_points(self):
        """Clear all captured objects and markers"""
        self.captured_objects = []
        self.cluster_results = []
        self.clearMarkers()
        self.clearClusterLayers()
        self.pointList = []
        self.transformedPoints = []
        self.iface.messageBar().pushSuccess("Cleared", "All captured points and markers cleared.")
        print("Cleared all captured points")


#Various functions
def write_to_csv(self, scriptPath, xcoord, ycoord, azimuth, altitude, declination, stars):
    global RESULTS_PATH
    if stars:
        starsString = ', '.join(stars)
    else:
        starsString = ''

    data = []
    data.append(ycoord)
    data.append(xcoord)
    data.append(azimuth)
    data.append(altitude)
    data.append(declination)
    data.append(starsString)

    if (starsString == ''):
        starsString = 'None'

    messageString = "The bearing of the line is: Az: " + str(azimuth) + " Alt: " + str(altitude) + " Declination: " + str(declination) + " pointing to: " + starsString 
    print(messageString)
    self.iface.messageBar().pushMessage("Result", messageString, Qgis.Info)

    if RESULTS_PATH == "Empty":
        RESULTS_PATH = os.getcwd()

    save = Ui_Save(data, RESULTS_PATH, os.path.join(scriptPath, "save_data.ui"))
    save.setWindowIcon(QtGui.QIcon(':/plugins/a2i/logo/icons/logo.png'))
    save.exec()
    
def zoom_to_coords():
    qid = QInputDialog()
    qid.setWindowIcon(QtGui.QIcon(logo_icon_path))
    canvas = iface.mapCanvas()
    input, ok = QInputDialog.getText( qid, "Enter Coordinates", "Enter New Coordinates as 'x.xxx, y.yyy'", QLineEdit.Normal, "lat" + "," + "long")
    if ok:
        y = input.split( "," )[ 0 ]
        #print (y)
        x = input.split( "," )[ 1 ]
        #print (x)
        while (y == "lat" or x == "long") and ok:
            input, ok = QInputDialog.getText( qid, "Enter Coordinates", "Enter New Coordinates as 'x.xxx,y.yyy'", QLineEdit.Normal, "lat" + "," + "long")
            if ok:
                y = input.split( "," )[ 0 ]
                x = input.split( "," )[ 1 ]
        if ok:
            point = QgsPointXY(float(x), float(y))
            tr = QgsCoordinateTransform(QgsCoordinateReferenceSystem(QGIS_CRS), QgsCoordinateReferenceSystem(TARGET_CRS), QgsProject.instance())
            transformed_point = tr.transform(point, QgsCoordinateTransform.ReverseTransform)
            x = transformed_point.x()
            y = transformed_point.y()
            if not x:
                print ("x value is missing!")
            if not y:
                print ("y value is missing!")
            scale=200
            #print(x)
            #print(y)
            rect = QgsRectangle(x-scale,y-scale,x+scale,y+scale)
            canvas.setExtent(rect)
            canvas.refresh()

def azimuth_tool():
    iface.mapCanvas().setMapTool( canvas_clicked )

def rmvLyr(lyrname):
    qinst = QgsProject.instance()
    qinst.removeMapLayer(qinst.mapLayersByName(lyrname)[0].id())

def set_params():

    change = False

    ui = Ui_Dialog(SCRIPT_PATH)
    ui.setWindowIcon(QtGui.QIcon(logo_icon_path))
    ui.exec()

    with open(config_path, 'r') as f:
        #SCRIPT_PATH = f.readline().rstrip("\n")
        f.readline()
        RESULTS_PATH = f.readline().rstrip("\n")
        PYTHON_PATH = f.readline().rstrip("\n")

        if f.readline().rstrip("\n") == "Yes":
            DOWNLOAD_MAP = True
            mapType = f.readline().rstrip("\n")
            
            if mapType == "Roadmap":
                if MAP_TYPE != "mt1.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}":
                    change = True
                MAP_TYPE = "mt1.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}"
            elif mapType == "Terrain":
                if MAP_TYPE != "mt1.google.com/vt/lyrs=p&hl=en&x={x}&y={y}&z={z}":
                    change = True
                MAP_TYPE = "mt1.google.com/vt/lyrs=p&hl=en&x={x}&y={y}&z={z}"
            elif mapType == "Satellite":
                if MAP_TYPE != "mt1.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}":
                    change = True
                MAP_TYPE = "mt1.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}"
            elif mapType == "Hybrid":
                if MAP_TYPE != "mt1.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}":
                    change = True
                MAP_TYPE = "mt1.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}"
            else:
                DOWNLOAD_MAP = False
                f.readline()

        LINE_WIDTH = float(f.readline().rstrip("\n"))
        SCRIPT_SLEEP = float(f.readline().rstrip("\n"))

    if change:
        rmvLyr("Google Sat")
        service_url = MAP_TYPE
        service_uri = "type=xyz&zmin=0&zmax=21&url=https://"+requests.utils.quote(service_url)
        iface.addRasterLayer(service_uri, "Google Sat", "wms")


#'Main' code, executed first when you run the script
def initA2i(scriptPath, iface):
    #Print empty line between console ouputs
    print('\n')
    SCRIPT_PATH = scriptPath

    #Set paths to useful files
    config_path = os.path.join(SCRIPT_PATH, "config.txt")
    ui_path = os.path.join(SCRIPT_PATH, "dialog.ui")
    save_path = os.path.join(SCRIPT_PATH, "save_data.ui")
    tool_icon_path = ':/plugins/a2i/toolbar/icons/bearing.png'
    location_icon_path = ':/plugins/a2i/toolbar/icons/location.png'
    params_icon_path = ':/plugins/a2i/toolbar/icons/settings.png'
    logo_icon_path = ':/plugins/a2i/logo/icons/logo.png'

    #Read config file
    with open(config_path, 'r') as f:
        #SCRIPT_PATH = f.readline().rstrip("\n")
        f.readline()
        RESULTS_PATH = f.readline().rstrip("\n")
        PYTHON_PATH = f.readline().rstrip("\n")

        if f.readline().rstrip("\n") == "Yes":
            DOWNLOAD_MAP = True
            mapType = f.readline().rstrip("\n")
            if mapType == "Roadmap":
                MAP_TYPE = "mt1.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}"
            elif mapType == "Terrain":
                MAP_TYPE = "mt1.google.com/vt/lyrs=p&hl=en&x={x}&y={y}&z={z}"
            elif mapType == "Satellite":
                MAP_TYPE = "mt1.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}"
            elif mapType == "Hybrid":
                MAP_TYPE = "mt1.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}"
            else:
                DOWNLOAD_MAP = False
                f.readline()

        LINE_WIDTH = float(f.readline().rstrip("\n"))
        SCRIPT_SLEEP = float(f.readline().rstrip("\n"))

    #Set the first too as the pan tool
    iface.actionPan().trigger()

    #Deal with map download and type
    if DOWNLOAD_MAP:
        service_url = MAP_TYPE
        service_uri = "type=xyz&zmin=0&zmax=21&url=https://"+requests.utils.quote(service_url)
        #print ("YES!")

        if QgsProject.instance().mapLayersByName("Google Sat"):
            print("Google image is already loaded!")
        else:
            iface.addRasterLayer(service_uri, "Google Sat", "wms")

    #Initialize the tool
    canvas_clicked = DeclinationTool(iface.mapCanvas(), iface)

    #Set the buttons on the toolbar
    action_tool = QAction(QIcon(tool_icon_path), 'Start Tool')
    action_tool.triggered.connect(azimuth_tool)
    iface.addToolBarIcon(action_tool)

    action_zoom = QAction(QIcon(location_icon_path), 'Go to Coords')
    action_zoom.triggered.connect(zoom_to_coords)
    iface.addToolBarIcon(action_zoom)

    set_parameters = QAction(QIcon(params_icon_path), 'Set Params')
    set_parameters.triggered.connect(set_params)
    iface.addToolBarIcon(set_parameters)

    print ('OK')



	  	