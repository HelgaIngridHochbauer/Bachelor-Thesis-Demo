#!/usr/bin/env python3
"""
Python version of script.R
Downloads horizon data from HeyWhatsThat.com and calculates altitude for a given azimuth.
"""

import sys
import os
import re
import csv
import math
import tempfile
import urllib.request
import urllib.parse


def download_hwt(hwt_id):
    """
    Download horizon data from HeyWhatsThat.com
    
    Args:
        hwt_id: 8-character HeyWhatsThat ID
        
    Returns:
        dict: Dictionary containing metadata and horizon data
    """
    if len(hwt_id) != 8:
        raise ValueError('Incorrect HeyWhatsThat ID.')
    
    hor = {}
    
    # Horizon metadata
    url = f"http://www.heywhatsthat.com/iphone/pan.cgi?id={hwt_id}"
    with urllib.request.urlopen(url) as response:
        html_content = response.read().decode('utf-8')
    
    # Lat/Lon/Elev
    pattern = r'<div class="details_data">([^<]*)</div>'
    matches = re.findall(pattern, html_content)
    
    if len(matches) < 4:
        raise ValueError("Could not parse metadata from HTML")
    
    # Extract latitude
    lat_str = matches[0]
    lat_match = re.search(r'([\d.]+)', lat_str)
    if lat_match:
        lat = float(lat_match.group(1))
        if 'S' in lat_str:
            lat = -lat
    else:
        raise ValueError("Could not parse latitude")
    
    # Extract longitude
    lon_str = matches[1]
    lon_match = re.search(r'([\d.]+)', lon_str)
    if lon_match:
        lon = float(lon_match.group(1))
        if 'W' in lon_str:
            lon = -lon
    else:
        raise ValueError("Could not parse longitude")
    
    # Extract elevation
    elev_str = matches[3]
    elev_match = re.search(r'([\d.]+)', elev_str)
    if elev_match:
        elev = float(elev_match.group(1))
    else:
        raise ValueError("Could not parse elevation")
    
    # Site Name
    name_pattern = r'<div id="pan_top_title" class="ellipsis" style="position: absolute; top: 46px; width: 296px">([^<]*)</div>'
    name_match = re.search(name_pattern, html_content)
    if name_match:
        name = name_match.group(1)
    else:
        name = "Unknown"
    
    # Horizon data
    temp_dir = tempfile.gettempdir()
    csv_file = os.path.join(temp_dir, f"{hwt_id}.csv")
    
    # Check if already downloaded
    existing_files = [f for f in os.listdir(temp_dir) if f.startswith(hwt_id) and f.endswith('.csv')]
    
    if not existing_files:
        # Clean up old files if too many
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv') and len(f) >= 8]
        if len(csv_files) > 500:
            # Get file modification times and delete oldest
            file_times = [(f, os.path.getmtime(os.path.join(temp_dir, f))) for f in csv_files]
            file_times.sort(key=lambda x: x[1])
            oldest_id = file_times[0][0][:8]
            for f in os.listdir(temp_dir):
                if f.startswith(oldest_id):
                    try:
                        os.remove(os.path.join(temp_dir, f))
                    except:
                        pass
        
        # Download new CSV
        csv_url = f'http://www.heywhatsthat.com/api/horizon.csv?id={hwt_id}&resolution=.125'
        urllib.request.urlretrieve(csv_url, csv_file)
    
    # Read horizon CSV
    horizon_data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            horizon_data.append(row)
    
    # Calculate altitude error
    delta = 9.73  # 9.73m for SRTM data
    
    # Find distance column name (could be 'distance (m)', 'distance..m.', etc.)
    if not horizon_data:
        raise ValueError("No horizon data found in CSV")
    
    distance_col = None
    for key in horizon_data[0].keys():
        if 'distance' in key.lower():
            distance_col = key
            break
    
    if distance_col is None:
        raise ValueError("Could not find distance column in CSV")
    
    for i, row in enumerate(horizon_data):
        hor_alt = float(row['altitude'])
        distance = float(row[distance_col])
        
        delta_elev = distance * math.tan(math.radians(hor_alt))
        aux0 = math.degrees(math.atan(delta_elev / distance))
        aux1 = math.degrees(math.atan((delta_elev + delta) / distance))
        aux2 = math.degrees(math.atan((delta_elev - delta) / distance))
        
        error = abs(aux1 - aux0) + abs(aux2 - aux0)
        error = error / 2.0  # mean of absolute differences
        
        row['error'] = error
    
    # Build result structure
    hor['metadata'] = {
        'ID': hwt_id,
        'name': name,
        'georef': {'Lat': lat, 'Lon': lon, 'Elev': elev},
        'elevation': elev
    }
    
    # Find bin.bottom column (could be 'bin.bottom', 'bin_bottom', etc.)
    bin_col = None
    for key in horizon_data[0].keys():
        if 'bin' in key.lower() and 'bottom' in key.lower():
            bin_col = key
            break
    
    if bin_col is None:
        # Try common variations
        for key in ['bin.bottom', 'bin_bottom', 'azimuth', 'az']:
            if key in horizon_data[0]:
                bin_col = key
                break
    
    if bin_col is None:
        raise ValueError("Could not find bin.bottom/azimuth column in CSV")
    
    hor['data'] = []
    for row in horizon_data:
        hor['data'].append({
            'az': float(row[bin_col]),
            'alt': float(row['altitude']),
            'alt.unc': row.get('error', 0.0)
        })
    
    return hor


def hor2alt(hor, az):
    """
    Interpolate altitude for a given azimuth from horizon data
    
    Args:
        hor: Horizon data dictionary
        az: Azimuth in degrees
        
    Returns:
        float: Altitude in degrees
    """
    # Prepare data for interpolation
    az_values = [d['az'] for d in hor['data']]
    alt_values = [d['alt'] for d in hor['data']]
    
    # Create extended arrays for wrapping (az-360, az, az+360)
    az_extended = [a - 360 for a in az_values] + az_values + [a + 360 for a in az_values]
    alt_extended = alt_values * 3
    
    # Interpolate
    if az < min(az_extended) or az > max(az_extended):
        # Find closest value
        closest_idx = min(range(len(az_extended)), key=lambda i: abs(az_extended[i] - az))
        alt = alt_extended[closest_idx]
    else:
        # Linear interpolation
        for i in range(len(az_extended) - 1):
            if az_extended[i] <= az <= az_extended[i + 1]:
                # Linear interpolation
                t = (az - az_extended[i]) / (az_extended[i + 1] - az_extended[i])
                alt = alt_extended[i] + t * (alt_extended[i + 1] - alt_extended[i])
                break
        else:
            # Fallback to closest value
            closest_idx = min(range(len(az_extended)), key=lambda i: abs(az_extended[i] - az))
            alt = alt_extended[closest_idx]
    
    return round(alt, 2)


def main():
    """Main function to process command line arguments and output result"""
    #print("now running script.py (no need for R)")
    if len(sys.argv) < 3:
        print("Usage: script.py <HWTID> <azimuth>", file=sys.stderr)
        sys.exit(1)
    
    hwt_id = sys.argv[1]
    azimuth = float(sys.argv[2])
    
    try:
        hor = download_hwt(hwt_id)
        output = hor2alt(hor, azimuth)
        print(output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

