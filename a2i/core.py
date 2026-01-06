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
import sys
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
from qgis.core import QgsLineSymbol
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
from .clustering import get_clustering_method, list_methods

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
        self.captured_objects = []  # List of tuples: ((point1, point2), (transformed1, transformed2), azimuth_or_none)
        # azimuth_or_none: float if provided/calculated, None if not yet calculated
        self.markers = []  # List of QgsVertexMarker objects for visualization
        self.cluster_results = []  # Store cluster calculation results
        self.cluster_layers = []  # List of QgsVectorLayer for cluster visualizations (not used anymore, kept for compatibility)
        self.clustering_method_name = 'dbscan'  # Default clustering method
        
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
                # Store the object (both points) and calculated azimuth
                object_points = (self.pointList[0], self.pointList[1])
                object_transformed = (self.transformedPoints[0], self.transformedPoints[1])
                self.captured_objects.append((object_points, object_transformed, self.az))
                
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
        
        # Validate paths before subprocess call
        if not os.path.exists(PYTHON_PATH):
            raise ValueError(f"Python executable not found: {PYTHON_PATH}")
        if not os.path.exists(script_path):
            raise ValueError(f"Script not found: {script_path}")
        
        args = [PYTHON_PATH, script_path, self.code, str(self.az)]
        result = ''

        # Use shell=False for better security (arguments are already in list format)
        # shell=True is not needed when passing a list of arguments
        # Suppress terminal window on Windows (CREATE_NO_WINDOW available in Python 3.7+)
        creation_flags = 0
        if sys.platform == 'win32':
            # 0x08000000 is CREATE_NO_WINDOW constant (available in Python 3.7+)
            creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
        result = subprocess.run(args, capture_output=True, shell=False, text=True, creationflags=creation_flags)
        i = 1
        while result.returncode != 0 and i <= 10:
            print("Horizon Python script running, please wait...")
            time.sleep(SCRIPT_SLEEP)
            result = subprocess.run(args, capture_output=True, shell=False, text=True, creationflags=creation_flags)
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
        for obj_data in self.captured_objects:
            # Unpack: (obj_points, obj_transformed, azimuth_or_none)
            if len(obj_data) == 3:
                obj_points, obj_transformed, azimuth_or_none = obj_data
            else:
                # Backward compatibility: old format without azimuth
                obj_points, obj_transformed = obj_data
                azimuth_or_none = None
            # Calculate midpoint (centroid) of the object
            p1 = obj_transformed[0]
            p2 = obj_transformed[1]
            centroid_lat = (p1.y() + p2.y()) / 2.0
            centroid_lon = (p1.x() + p2.x()) / 2.0
            object_centroids.append((centroid_lat, centroid_lon, obj_points, obj_transformed, azimuth_or_none))
        
        # Allow Qt to process events
        QCoreApplication.processEvents()
        
        # Use the selected clustering method
        distance_threshold = 100.0  # 100 meters
        
        try:
            clustering_method = get_clustering_method(self.clustering_method_name)
            labels = clustering_method.cluster(object_centroids, distance_threshold=distance_threshold)
        except Exception as e:
            print(f"Error in clustering method '{self.clustering_method_name}': {e}")
            self.iface.messageBar().pushWarning("Warning", f"Clustering failed: {e}")
            return []
        
        # Allow Qt to process events after clustering
        QCoreApplication.processEvents()
        
        # Validate labels length matches input data
        if len(labels) != len(object_centroids):
            error_msg = f"Clustering method returned {len(labels)} labels for {len(object_centroids)} objects. Lengths must match."
            print(f"Error: {error_msg}")
            self.iface.messageBar().pushWarning("Warning", f"Clustering failed: {error_msg}")
            return []
        
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
            # Protect against division by zero (shouldn't happen, but safer)
            num_objects_in_cluster = len(objects)
            if num_objects_in_cluster == 0:
                continue  # Skip empty clusters
            mean_lat = sum(obj[0] for obj in objects) / num_objects_in_cluster
            mean_lon = sum(obj[1] for obj in objects) / num_objects_in_cluster
            
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
    
    def select_clustering_method(self):
        """Open a dialog to select the clustering method"""
        available_methods = list_methods()
        method_descriptions = []
        method_name_map = {}  # Map description to method name
        
        for method_name in available_methods:
            try:
                method = get_clustering_method(method_name)
                description = f"{method.name}: {method.description}"
                method_descriptions.append(description)
                method_name_map[description] = method_name
            except Exception as e:
                print(f"Warning: Could not load method {method_name}: {e}")
                method_descriptions.append(method_name)
                method_name_map[method_name] = method_name
        
        # Find current selection index
        current_index = 0
        if self.clustering_method_name in available_methods:
            try:
                current_method = get_clustering_method(self.clustering_method_name)
                current_description = f"{current_method.name}: {current_method.description}"
                if current_description in method_descriptions:
                    current_index = method_descriptions.index(current_description)
            except:
                if self.clustering_method_name in method_descriptions:
                    current_index = method_descriptions.index(self.clustering_method_name)
        
        item, ok = QInputDialog.getItem(
            self.iface.mainWindow(),
            "Select Clustering Method",
            "Choose a clustering algorithm:",
            method_descriptions,
            current_index,
            False
        )
        
        if ok and item:
            # Map description back to method name
            if item in method_name_map:
                self.clustering_method_name = method_name_map[item]
                try:
                    method = get_clustering_method(self.clustering_method_name)
                    self.iface.messageBar().pushInfo("Clustering Method", f"Selected: {method.name}")
                    print(f"Clustering method set to: {self.clustering_method_name} ({method.name})")
                except:
                    self.iface.messageBar().pushInfo("Clustering Method", f"Selected: {self.clustering_method_name}")
                    print(f"Clustering method set to: {self.clustering_method_name}")
    
    def calculate_cluster_orientation(self, cluster_info, progress=None):
        """Calculate orientation and declination for a cluster centroid"""
        centroid_lat = cluster_info['centroid_lat']
        centroid_lon = cluster_info['centroid_lon']
        objects = cluster_info['objects']
        
        # Calculate average azimuth from all objects in cluster
        # Use stored azimuth if available, otherwise calculate from points
        azimuths = []
        for obj in objects:
            # obj format: (centroid_lat, centroid_lon, obj_points, obj_transformed, azimuth_or_none)
            if len(obj) >= 5 and obj[4] is not None:
                # Use stored azimuth
                az = obj[4]
            else:
                # Calculate azimuth from points
                obj_points = obj[2]
                az = computeAzimuth([obj_points[0], obj_points[1]])
            azimuths.append(az)
        
        # Use mean azimuth (or could use median for robustness)
        # Protect against division by zero
        if not azimuths:
            mean_az = 0
        else:
            mean_az = sum(azimuths) / len(azimuths)
        
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
                # Use shell=False for security - arguments are controlled and validated
                # Suppress terminal window on Windows (CREATE_NO_WINDOW available in Python 3.7+)
                creation_flags = 0
                if sys.platform == 'win32':
                    # 0x08000000 is CREATE_NO_WINDOW constant (available in Python 3.7+)
                    creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
                result = subprocess.run(args, capture_output=True, shell=False, text=True, timeout=30, creationflags=creation_flags)
                
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
            
            # Process clusters - this may take time, but clustering methods now process events internally
            clusters = self.process_clusters()
            
            # Process events immediately after clustering
            QCoreApplication.processEvents()
            
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
        
        # Save results (save function handles empty RESULTS_PATH)
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
                    # obj format: (centroid_lat, centroid_lon, obj_points, obj_transformed, azimuth_or_none)
                    obj_points = obj[2]
                    obj_transformed = obj[3]
                    # Get midpoint of object
                    obj_lat = (obj_transformed[0].y() + obj_transformed[1].y()) / 2.0
                    obj_lon = (obj_transformed[0].x() + obj_transformed[1].x()) / 2.0
                    
                    # Use stored azimuth if available, otherwise calculate
                    if len(obj) >= 5 and obj[4] is not None:
                        az = obj[4]
                    else:
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
    
    def import_from_csv(self, filepath):
        """
        Import objects from a CSV file.
        
        CSV Format (one row per object):
        - Required columns: lat1, lon1, lat2, lon2 (decimal degrees, EPSG:4326)
        - Optional column: azimuth (degrees, 0-360)
        
        Example CSV:
        lat1,lon1,lat2,lon2,azimuth
        45.1234,-73.5678,45.1240,-73.5680,45.5
        45.1300,-73.5700,45.1305,-73.5705,46.2
        
        If azimuth is not provided, it will be calculated from the points.
        """
        try:
            import csv as csv_module
            from qgis.PyQt.QtWidgets import QFileDialog
            
            # If no filepath provided, show file dialog
            if not filepath:
                global RESULTS_PATH
                if RESULTS_PATH == "Empty":
                    RESULTS_PATH = os.getcwd()
                
                filepath, _ = QFileDialog.getOpenFileName(
                    None,
                    "Import Objects from CSV",
                    RESULTS_PATH,
                    "Comma Separated Values Files (*.csv)"
                )
                
                if not filepath:
                    return False  # User cancelled
            
            # Read CSV file
            imported_count = 0
            errors = []
            
            # Cache coordinate transforms
            tr_to_target = QgsCoordinateTransform(
                QgsCoordinateReferenceSystem(QGIS_CRS),
                QgsCoordinateReferenceSystem(TARGET_CRS),
                QgsProject.instance()
            )
            
            with open(filepath, 'r', encoding='utf-8') as f:
                # Try to detect header
                reader = csv_module.reader(f)
                rows = list(reader)
                
                if not rows:
                    self.iface.messageBar().pushWarning("Import Error", "CSV file is empty.")
                    return False
                
                # Detect if first row is header
                header = rows[0]
                start_row = 0
                
                # Try to find column indices
                lat1_idx = lon1_idx = lat2_idx = lon2_idx = azimuth_idx = None
                
                # Check if header exists (non-numeric values)
                try:
                    float(header[0])
                    # No header, use indices 0-4
                    lat1_idx, lon1_idx, lat2_idx, lon2_idx = 0, 1, 2, 3
                    if len(header) >= 5:
                        azimuth_idx = 4
                    start_row = 0
                except (ValueError, IndexError):
                    # Header exists, find column names
                    header_lower = [col.lower().strip() for col in header]
                    for idx, col in enumerate(header_lower):
                        if col in ['lat1', 'latitude1', 'lat_1']:
                            lat1_idx = idx
                        elif col in ['lon1', 'longitude1', 'lon_1', 'lng1']:
                            lon1_idx = idx
                        elif col in ['lat2', 'latitude2', 'lat_2']:
                            lat2_idx = idx
                        elif col in ['lon2', 'longitude2', 'lon_2', 'lng2']:
                            lon2_idx = idx
                        elif col in ['azimuth', 'az', 'bearing']:
                            azimuth_idx = idx
                    start_row = 1
                
                # Validate required columns found
                if None in [lat1_idx, lon1_idx, lat2_idx, lon2_idx]:
                    self.iface.messageBar().pushCritical(
                        "Import Error",
                        "CSV must contain columns: lat1, lon1, lat2, lon2 (or latitude1/longitude1, etc.)"
                    )
                    return False
                
                # Process data rows
                for row_idx, row in enumerate(rows[start_row:], start=start_row + 1):
                    try:
                        if len(row) < max(lat1_idx, lon1_idx, lat2_idx, lon2_idx) + 1:
                            errors.append(f"Row {row_idx}: Not enough columns")
                            continue
                        
                        # Parse coordinates
                        lat1 = float(row[lat1_idx])
                        lon1 = float(row[lon1_idx])
                        lat2 = float(row[lat2_idx])
                        lon2 = float(row[lon2_idx])
                        
                        # Validate coordinate ranges
                        if not (-90 <= lat1 <= 90) or not (-90 <= lat2 <= 90):
                            errors.append(f"Row {row_idx}: Latitude out of range (-90 to 90)")
                            continue
                        if not (-180 <= lon1 <= 180) or not (-180 <= lon2 <= 180):
                            errors.append(f"Row {row_idx}: Longitude out of range (-180 to 180)")
                            continue
                        
                        # Parse optional azimuth
                        azimuth = None
                        if azimuth_idx is not None and azimuth_idx < len(row):
                            try:
                                azimuth_str = row[azimuth_idx].strip()
                                if azimuth_str:
                                    azimuth = float(azimuth_str)
                                    # Normalize to 0-360 range
                                    while azimuth < 0:
                                        azimuth += 360
                                    while azimuth >= 360:
                                        azimuth -= 360
                            except (ValueError, IndexError):
                                pass  # Azimuth not provided or invalid, will calculate
                        
                        # Create QgsPointXY objects in EPSG:4326 (target CRS)
                        point1_transformed = QgsPointXY(lon1, lat1)
                        point2_transformed = QgsPointXY(lon2, lat2)
                        
                        # Transform to map CRS (EPSG:3857) for visualization
                        point1_map = tr_to_target.transform(
                            point1_transformed,
                            QgsCoordinateTransform.ReverseTransform
                        )
                        point2_map = tr_to_target.transform(
                            point2_transformed,
                            QgsCoordinateTransform.ReverseTransform
                        )
                        
                        # Calculate azimuth if not provided
                        if azimuth is None:
                            azimuth = computeAzimuth([point1_map, point2_map])
                        
                        # Store object (same format as manual capture)
                        object_points = (point1_map, point2_map)
                        object_transformed = (point1_transformed, point2_transformed)
                        self.captured_objects.append((object_points, object_transformed, azimuth))
                        
                        # Add visualization markers and line
                        self.addObjectMarkers(object_points)
                        self.drawLineBetweenPoints(point1_map, point2_map)
                        
                        imported_count += 1
                        
                    except (ValueError, IndexError) as e:
                        errors.append(f"Row {row_idx}: {str(e)}")
                        continue
                
                # Report results
                if imported_count > 0:
                    self.iface.messageBar().pushSuccess(
                        "Import Complete",
                        f"Imported {imported_count} object(s) from CSV. Total objects: {len(self.captured_objects)}"
                    )
                    print(f"Imported {imported_count} object(s) from CSV file: {filepath}")
                    
                    if errors:
                        error_msg = f"({len(errors)} row(s) had errors - see console)"
                        print(f"Import errors: {errors}")
                        self.iface.messageBar().pushWarning("Import Warnings", error_msg)
                    
                    return True
                else:
                    self.iface.messageBar().pushWarning(
                        "Import Failed",
                        "No valid objects imported. Check CSV format."
                    )
                    if errors:
                        print(f"Import errors: {errors}")
                    return False
                    
        except Exception as e:
            error_msg = f"Error importing CSV: {str(e)}"
            self.iface.messageBar().pushCritical("Import Error", error_msg)
            print(f"CSV import error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def drawLineBetweenPoints(self, point1, point2):
        """Draw a line between two points on the canvas"""
        # Create or get line layer
        line_layer_name = "Imported Objects Lines"
        layer = None
        layers = QgsProject.instance().mapLayersByName(line_layer_name)
        
        if layers:
            layer = layers[0]
        else:
            # Create new layer
            layer = QgsVectorLayer("LineString?crs=" + QGIS_CRS, line_layer_name, "memory")
            QgsProject.instance().addMapLayer(layer)
            
            # Style the layer
            try:
                symbol = QgsLineSymbol.createSimple({
                    'line_color': '255,0,0,255',
                    'line_width': str(LINE_WIDTH)
                })
                layer.renderer().setSymbol(symbol)
            except:
                # Fallback: set width directly
                try:
                    layer.renderer().symbol().setWidth(LINE_WIDTH)
                except:
                    pass
        
        # Add feature
        provider = layer.dataProvider()
        feat = QgsFeature()
        geom = QgsGeometry.fromPolylineXY([point1, point2])
        feat.setGeometry(geom)
        provider.addFeature(feat)
        layer.updateExtents()


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



	  	