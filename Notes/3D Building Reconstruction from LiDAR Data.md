Scenario: You are a real estate investor evaluating the neighborhood's potential. You want to determine the feasibility of extending existing houses horizontally and vertically. To make informed decisions, you need precise information about each building's footprint and height. 

Python environment preparation --> Data Preparation --> Single building experiments --> Automation and Scaling

###### Phase 1: 3D Python Setup
```
# Three base libraries (pip install numpy matplotlib pandas)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3D libraries (pip install open3d)
import open3d as o3d

# To manipulate 3D point clouds (pip install laspy[lazrs, laszip])
import laspy

# Geospatial libraries (pip install rasterio)
import rasterio

# Geospatial data processing (pip install geopandas)
import geopandas as gpd

# To handle 2D vector data (pip install shapely alphashape)
import shapely as sh
import alphashape as ash

from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import Polygon
```

###### Phase 2: Data Preparation
```
# Neighborhood point cloud
las = laspy.read("../DATA/neighborhood.laz")

# Explore the classification field
print(np.unique(las.classification))
# What information does each point actually carry?
print([dimension.name for dimension in las.point_format.dimensions])

# Explore CRS (Coordinate Reference System) information
# Where in the real world is this building?
crs = las.vlrs[2].string
print(las.vlrs[2].string)
```

- To perform targeted semantic filter to isolate buildings from the rest of the neighborhood scan.

```
# Data Preprocessing

# Create a mask to filter points
# Under the standard ASPRS (American Society for Photogrammetry and Remote Sensing) classification codes used in LiDAR data, class code 6 explicitly represents buildings (roof sfaces and structural envelopes)
# This line creates a boolean mask (an array of True and False values) checking every single point in the file to see if it belongs to a building
pts_mask = las.classification == 6

# This uses that mask to pull out only the X, Y, and Z coordinates of the building points
# np.vstack stacks them into a NumPy array with a shape of  (3, N) (where the 3 rows are X, Y, and Z, and N is the total number of building points)
xyz_t = np.vstack((las.x[pts_mask], las.y[pts_mask], las.z[pts_mask]))

# Transform to Open3D.o3d.geometry.PointCloud and visualize
pcd_o3d = o3d.geometry.PointCloud()
# Open3D doesn't read a (3, N) shape; it expects points formatted as (N, 3) where every row is a single point containing [x, y, z]
# .transpose() flips your array axis so the shape matches what Open3D requires, and it assigns those points to an empty Open3D point cloud object
pcd_o3d.points = o3d.utility.Vector3dVector(xyz_t.transpose())

# Translate the point cloud, and keep the translation to reapply at the end
# We are shifting the entire point cloud so that its geometric center sits right at the world origin (0, 0, 0)
pcd_center = pcd_o3d.get_center()
# Subtracts the center value from every single point, sliding the entire building over so its middle aligns with (0, 0, 0)
pcd_o3d.translate(-pcd_center)

# Transform to Open3D.o3d.geometry.PointCLoud and visualize
o3d.visualization.draw_geometries([pcd_o3d])
```

```
# Isolating ground points
pts_mask = las.classification == 2

xyz_t = np.vstack((las.x[pts_mask], las.y[pts_mask], las.z[pts_mask]))

ground_pts = o3d.geometry.PointCloud()
ground_pts.points = o3d.utility.Vector3dVector(xyz_t.transpose())
ground_pts.translate(-pcd_center)

# Visualize the results
o3d.visualization.draw_geometries([ground_pts])

# Identifying the average distance between building points
# Because LiDAR density can vary wildly depending on the scanner, we comput the average nn_distance so our script dynamically knows: "Ah, the points in this specific file are spaced about 0.02 meters apart on average. Therefore, when I run my feature-detection or clustering algorithms, I should set my search radius to 0.04 meters"
nn_distance = np.mean(pcd_o3d.compute_nearest_neighbor_distance())
print("average point distance (m): ", nn_distance)
```

###### Phase 3: Experiments
Unsupervised segmentation --> House Isolation --> 2D building footprint --> Semantic extraction --> From 2D to 3D vectors --> 3D mesh modelling

**Unsupervised Point Cloud Segmentation**
```
# Definition of parameters epsilon, and the minimum number of points to be considered
# epsilon - the maximum distance search radius (in meters, since the data is in metric)
# "If a point is within 2 meters of another point, consider them part of the same neighborhood"
epsilon = 2
# min_cluster_points - density threshold
# "A group of points only count as a valid object (like a building or a structure) if it has at least 100 points clumped together. Anything smaller gets thrown away."
min_cluster_points = 100

# DBSCAN assigns an integer ID (a label) to every single point
# Points belonging to Building #1 might get labeled 0
# Points belonging to Building #2 might get labeled 1
# Stray points or noise that don't fit anywhere get labeled -1
labels = np.array(pcd_o3d.cluster_dbscan(eps=epsilon, min_points=min_cluster_points))
# max_label tells you how many distinct clusters (buildings / objects) the algorithm successfully found
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

# We use a discrete color palette to randomize the visualization
# Since point cloud doesn't come with pre-assigned colors for every separate building, this code uses Matplotlib's tab20 color palette to randomize a distinct color for each cluster
colors = plt.get_cmap("tab20") (labels / (max_label if max_label > 0 else 1))
# Any point labeled -1 (noise / stray debris) gets painted black or hidden so it doesn't distract you
colors[labels < 0] = 0
# This wraps those colors back into Open3D so that when you visualize it, every single building is cleanly color-coded
pcd_o3d.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualize
p3d.visulization.draw_geometries([pcd_o3d])
```

**3D House Segment Isolation**
```
# Selecting a segment to be considered
sel = 1
segment = pcd_o3d.select_by_index(np.where(labels==sel)[0])

o3d.visualization.draw_geometries([segment])
```

**2D Building Footprint Extraction**
- Hull - the outer boundary, skin, or envelope that encloses a group of objects. 
- Convex Hull - Always curving outwards (like a circle or a simple triangle)
- Concave Hull - Allows the shape to cave inward to match complex floor plans
```
# We extract only the X and Y coordinates of our point cloud
# Since building footprint is a 2D drawing (a plan view), we don't care about the roof height, ceiling elevations, or multi-storey vertical variations when drawing the ground perimeter. We slice away the Z (height) axis and keep only X and Y, flattening the 3D building down to a 2D floor plan
points_2D = np.asarray(segment.points)[:, 0:2]

# We compute the shape with alphashape and return the result with shapely
# Alpha Shape is a concave hull - by tuning the parameter, you give the algorithm permission to curve inward and hug the actual concave corners, re-creating the true architectural shape of the building's exterior walls (including setbacks, wings, and recesses)
building_vector = ash.alphashape(points_2D, alpha=0.5)
building_vector

# Store the 2D polygon in a Geodataframe
# Right now, the shape is just a mathematical object in Python memory. By wrapping it in a GeoPandas GeoDataFrame and attaching a Coordinate Reference System (crs), you turn it into a standard geospatial vector file
building_gdf = gpd.GeoDataFrame(geometry=[building_vector], crs='EPSG:26910')
building_gdf.head(1)
```

**Semantic and Attribute Extraction**
```
# Height of the building as a relative measure
# What is wrong with this line?
# It takes all the building points, grabs their Z coordinates ([:, 2]), adds back the global center offset (+ pcd_center[2]) to restore real-world elevation, and subtracts the minimum altitude from the maximum altitude
# Why it fails - it assumes the lowest point in the building point cloud is the ground. If data is missing at the base or there's a porch overhang, your minimum point is floating in mid-air, making your building look shorter than it actually is
altitude = np.asarray(segment.points)[:, 2] + pcd_center[2]
height_test = np.max(altitude) - np.min(altitude)
print('Is this correct: ', height_test)

# We first have to define the ground level in our local data
query_point = segment.get_center()
# It takes the center of the building, but deliberately overwrites the Z coordinate (query_point[2]) with the lowest vertical boundary of the building (get_min_bound()[2])
query_point[2] = segment.get_min_bound()[2]
# This builds a lightning fast k-d tree index out of all the ground points in the neighborhood. This allows us to query the ground spatially without scanning every single point manually
pcd_tree = o3d.geometry.KDTreeFlann(ground_pts)
It uses our anchor query_point at the base of the building to search the ground_tree, grabbing the 200 closest ground points immediately surrounding that specific building base
[k, idx, _] = pcd_tree.search_knn_vector_3d(query_point, 200)

# From the nn search, we extract the points that belong to the ground and paint them gray
sample = ground_pts.select_by_index(idx, invert=False)
sample.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([sample, ground_pts])

# Extract the mean value of the ground in this specific place
# Instead of trusting a single random minimum point, it averages the Z elevation of all 200 surrounding ground points. This gives you a stable, reliable local grade elevation (ground_zero), even if the site is sloped or uneven
ground_zero = sample.get_center()[2]

# Compute the true height of the building, roof included
# It takes the absolute highest point of the roof (segment.get_max_bound()[2]) and subtracts the true ground_zero elevation
height = segment.get_max_bound()[2] - ground_zero
print('True Height: ', height)

# Check the difference
print('Height Difference: ', height - height_test)
```

```
# Computing parameters
building_gdf[['id']] = sel
building_gdf[['height']] = segment.get_max_bound()[2] - sample.get_center()[2]
building_gdf[['area']] = building_vector.area
building_gdf[['perimeter']] = building_vector.length
building_gdf[['local_cx', 'local_cy', 'local_cz']] = np.asarray([building_vector.centroid.x, building_vector.centroid.y, sample.get_center()[2]])
building_gdf[['transl_x', 'transl_y', 'transl_z']] = pcd_center
building_gdf[['pts_number']] = len(segment.points)

building_gdf.head(1)
```

```
# Extra attributes
# You are plotting a histogram of the Z valies to look at the vertical profile of the building. 
points_1D = np.asarray(segment.points)[:, 2]
print('The local minima (along the Z axis)', np.min(points_1D))
print('The local maxima (along the Z axis)', np.max(points_1D))

# Bins group your data into ranges (intervals)
# Histogram counts how many building points fall into each slot and draws a bar showing that count
plt.hist(points_1D, bins='auto')
plt.title("Histogram with 'auto' bins")
plt.show()

o3d.visualization.draw_geometries([segment])
```

**2D to 3D Vectors**
```
# The base layer
# Generate the vertices list
vertices = list(building_vector.exterior.coords)

# Construct the Open3D object
# It grabs all the corner coordinates (x, y) of your 2D building polygon and appends a 0 for the Z coordinate (point + (0,)), placing the entire outline flat on the ground
polygon_2d = o3d.geometry.LineSet()
polygon_2d.points = o3d.utility.Vector3dVector(
	[point + (0,) for point in vertices]
)

# Vector2iVector connects corner i to corner i + 1 in a loop, drawing crisp lines between all the vertices to form a wireframe perimeter
polygon_2d_lines = o3d.utility.Vector2iVector(
	[(i, (i + 1) % len(vertices)) for i in range(len(vertices))]
)

o3d.visualization.draw_geometries([polygon_2d])
```

```
# The top layer
# Generate the same element for the extruded
extrusion = o3d.geometry.LineSet()
# It takes those exact same 2D corner coordinates, but instead of adding 0 for Z, it adds your calculated height (point + (height,))
extrusion_points = o3d.utility.Vector3dVector(
	[point + (height,) for point in vertices]
)
extrusion_lines = o3d.utility.Vector2iVector(
	[(i, (i + 1) % len(vertices)) for i in range(len(vertices))]
)

o3d.visualization.draw_geometries([polygon_2d, extrusion])

# Plot the vertices
temp = polygon_2d + extrusion
temp.points
temp_o3d = o3d.geometry.PointCloud()
temp_o3d.points = temp.points

o3d.visualization.draw_geometries([temp_o3d])
```

```
# Generating the base vertices for the 3D Mesh with NumPy
# Pulls all the (x, y) corner coordinates of the building's polygon boundary into a NumPy array. If your building footprint has 10 corners, a is a matrix with a shape of (10, 2)
a = np.array(building_vector.exterior.coords)

# b creates a column of numbers matching the exact ground elevation (sample.get_center()[2]) for every single corner
b = np.ones([a.shape[0], 1]) * sample.get_center()[2]
# c creates a column matching the exact roof elevation (ground_zero + height) for every corner
c = np.ones([a.shape[0], 1]) * (sample.get_center()[2] + height)

# Define the ground footprint and the height arrays of points
# np.hstack (horizontal stack) glues your (x, y) coordinates (a) together with your Z elevation column (b or c)
# Now you have two complete 3D point sets: ground_pc (the ring of points resting flat on the ground) and up_pc (the matching ring of points floating at roof level)
ground_pc = np.hstack((a, b))
up_pc = np.hstack((a, c))

# Generate an open3d point cloud made of the major points
temp_o3d = o3d.geometry.PointCloud()
temp_o3d.points = o3d.utility.Vector3dVector(np.concatenate((ground_pc, up_pc), axis=0))

o3d.visualization.draw_geometries([temp_o3d])
```

**3D Model Creation: Mesh**
```
# Compute the alpha shape of the 3D base points
alpha = 20

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(temp_o3d, alpha)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.5, 0.4, 0])

o3d.visualization.draw_geometries([temp_o3d, mesh, segment], mesh_show_back_face=True)

# Repositioning the mesh from local to the world coordinates
mesh.translate(pcd_center)

# Export the mesh
import os
os.makedirs("../RESULTS/", exist_ok=True)
o3d.io.write_triangle_mesh('../RESULTS/house_sample.ply', mesh, write_ascii=False, compressed=True, write_vertex_normals=False, write_vertex_colors=False, wrote_triangle_uvs=False)

building_gdf.to_file("../RESULTS/single_building.shp")
```

###### Phase 4: Automation and Scaling
```
def random_color_generator():
	r = random.randint(0, 255)
	g = random.randint(0, 255)
	b = random.randint(0, 255)
	return [r/255, g/255, b/255]

# Initializing the GeodataFrame
buildings_gdf = gpd.GeoDataFrame(
	columns=['id', 'geometry', 'height', 'area', 'perimeter', 'local_cx', 'local_cy', 'local_cz', 'transl_x', 'transl_y', 'transl_z'], geometry='geometry', crs='EPSG:26910'
	)

# Reducing the output wave of Open3D
# When you loop through hundreads of buildings, Open3D likes to spam the terminal with minor warnings. This like turns off the noise so your console can stay clean
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

#Creating the loop
for sel in range(max_label+1):
	#1. Select the Segment
	segment = pcd_o3d.select_by_index(np.where(labels==sel)[0])

	#2. Compute the building footprint
	points_2D = np.asarray(segment.points)[:,0:2]
	building_vector = ash.alphashape(points_2D, alpha=0.5)

	#3. Compute the height of the segment (house candidate).
	query_point = segment.get_center()
	query_point[2] = segment.get_min_bound()[2]
	pcd_tree = o3d.geometry.KDTreeFlann(ground_pts)
	[k, idx, _] = pcd_tree.search_knn_vector_3d(query_point, 50)
	sample = ground_pts.select_by_index(idx, invert=False)
	ground_zero = sample.get_center()[2]
	height = segment.get_max_bound()[2] - ground_zero

	#4. Create the geopandas with attributes entry
	building_gdf = gpd.GeoDataFrame(geometry=[building_vector], crs='EPSG:26910')

	building_gdf[['id']] = sel
	building_gdf[['height']] = segment.get_max_bound()[2] - sample.get_center()[2]
	building_gdf[['area']] = building_vector.area
	building_gdf[['perimeter']] = building_vector.length
	building_gdf[['local_cx','local_cy','local_cz']] = np.asarray([building_vector.centroid.x, building_vector.centroid.y, sample.get_center()[2]])
	building_gdf[['transl_x','transl_y','transl_z']] = pcd_center
	building_gdf[['pts_number']] = len(segment.points)

	#4. Add it to geometries entries
	buildings_gdf = pd.concat([buildings_gdf, building_gdf])

	#5. Compute the 3D Vertices Geometries
	a = np.array(building_vector.exterior.coords)
	b = np.ones([a.shape[0],1])*sample.get_center()[2]
	c = np.ones([a.shape[0],1])*(sample.get_center()[2] + height)
	ground_pc = np.hstack((a, b))
	up_pc = np.hstack((a, c))
	temp_o3d = o3d.geometry.PointCloud()
	temp_o3d.points = o3d.utility.Vector3dVector(
		np.concatenate((ground_pc, up_pc), axis=0)
	)

	#5. Compute the 3D Geometry of a house
	alpha = 20
	mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(temp_o3d, alpha)
	mesh.translate(pcd_center)
	mesh.paint_uniform_color(random_color_generator()) 
	
	
	o3d.io.write_triangle_mesh('../RESULTS/house_'+str(sel)+'.ply', mesh, write_ascii=False, compressed=True, write_vertex_normals=False)
```

```
# Raster Variant
pixel_size = 1

x, y, z = xyz_t[0], xyz_t[1], xyz_t[2]

# Determine the extent of the DEM
min_x, max_x = np.min(x), np.max(x)
min_y, max_y = np.min(y), np.max(y)
# Calculate the number of pixels in X and Y directions
num_pixels_x = int((max_x - min_x) / pixel_size)
num_pixels_y = int((max_y - min_y) / pixel_size)

print("number of pixel along the X and Y: ", num_pixels_x, num_pixels_y)

# Create a transformation for the GeoTIFF
transform = from_origin(min_x, max_y, pixel_size, pixel_size)

# Create an array to store the elevation values
dem_array = np.zeros((num_pixels_y, num_pixels_x), dtype=np.float32)

# Convert X, Y coordinated to pixel indices
col_indices = ((x - min_x) / pixel_size).astype(int)
row_indices = ((max_y - y) / pixel_size).astype(int)

# Mask to ensure indices are within bounds
valid_indices = (0 <= row_indices) & (row_indices < num_pixels_y) & (0 <= col_indices) & (col_indices < num_pixels_x)

# Populate the DEM array with elevation values from the point cloud
dem_array[row_indices[valid_indices], col_indices[valid_indices]] = z[valid_indices]

# Save the DEM as a GeoTIFF file
with rasterio.open("../RESULTS/output_dem.tif", 'w', driver='GTiff', height=num_pixels_y, width=num_pixels_x, count=1, dtype=np.float32, crs='EPSG:26910', transform=transform) as dst: dst.write(dem_array, 1)

mask = None
with rasterio.Env():
	with rasterio.open("../RESULTS/output_dem.tif") as src:
		image = src.read(1) # first band
		results = (
		{'properties': {'raster_val': v}, 'geometry': s}
		for i, (s, v)
		in enumerate(
		shapes(image.astype(np.float32), mask=mask, transform=src.transform))
		)

geoms = list(results)

gdf = gpd.GeoDataFrame.from_features(geoms)
```