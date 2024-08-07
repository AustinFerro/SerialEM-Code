"""
Created on Tue Jul 23 11:28:08 2024

@author: Austin Ferro

Dependencies = caveclient 
fcl (pip install python-fcl)
meshparty 
see below 
"""

from caveclient import CAVEclient
from meshparty import trimesh_io, trimesh_vtk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trimesh import collision
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError
import trimesh
import json
from scipy.spatial import cKDTree
import time
from datetime import datetime


def main():

    start = time.time()
    start_datetime = datetime.fromtimestamp(start)
    
    print("Cell_Colli started at:",start_datetime)
    
    base_directory = "D:\MicronsEMdata" #change to whatever base directory you would like to save data to
    
    print("Script started")
    # MINNIE
    client = CAVEclient()
    client = CAVEclient('minnie65_public_v117')
    mm = trimesh_io.MeshMeta(cv_path=client.info.segmentation_source(),
                              disk_cache_path='minnie65_v117_meshes',map_gs_to_https=True)
    
    # Initialize variables for processing
    scale = np.array([4,4,40])
    voxel_size = np.array([8, 8, 80]) #allows for collision points to be segmented together within the size of the voxel
    microglia_points = {} #sets up library of all microglial interactions with the given OPC to be called for visualization

    
    #function to merge collision points based off of distance from one another
    def merge_voxels(voxels, voxel_size):
        merged_voxels = {}
        visited = set()
    
        def find_adjacent(voxel_key, adj_range=1): # Range is how close a voxel needs to be in order to be called as a part of a larger whole or an individual collision
            x, y, z = voxel_key
            for dx in range(-adj_range, adj_range + 1):
                for dy in range(-adj_range, adj_range + 1):
                    for dz in range(-adj_range, adj_range + 1):
                        yield (x + dx, y + dy, z + dz)
    
        # Wrap your outer loop with tqdm for a progress bar
        voxel_keys = list(voxels.keys())
        for key in tqdm(voxel_keys, desc="Merging voxels"):
            if key not in visited:
                stack = [key]
                group_points = []   
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    group_points.extend(voxels[current])
                    for adjacent in find_adjacent(current, adj_range=50):
                        if adjacent in voxels and adjacent not in visited:
                            stack.append(adjacent)
                if group_points:
                    group_array = np.array(group_points)
                    centroid = group_array.mean(axis=0)
                    merged_voxels[key] = {'points': group_array, 'centroid': centroid}
        return merged_voxels
    
    def calculate_surface_area(points):
        try:
            hull = ConvexHull(points)
            return hull.area  # This area is now in real-world units
        except QhullError:  # Use the correctly imported exception
            return 0
    
    def unique_points(points):
        # This line ensures that all points are unique by rounding (to handle floating point precision issues) and then finding unique rows.
        unique_p = np.unique(np.round(points, decimals=7), axis=0)
        return unique_p
    
    def create_sphere_meshes(centers, radius, c):
        """
        Create mesh spheres for each center point and return them as vtk actors.
        
        :param centers: Array of center points (each an array or list [x, y, z])
        :param radius: Radius of the spheres
        :return: List of vtk actors for each sphere
        """
        sphere_actors = []
        for center in centers:
            # Create a sphere mesh at each center
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            sphere.apply_translation(center)
            
            # Convert the trimesh sphere to a vtk actor
            sphere_actor = trimesh_vtk.mesh_actor(sphere, color=c, opacity=.75)
            sphere_actors.append(sphere_actor)
        
        return sphere_actors
    
    def load_PLs(file_path):
        # Load JSON data from a file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Ensure data contains the expected structure
        if 'annotations' in data:
            # Extract every "point" triplet and convert each to a tuple
            points = tuple(tuple(item['point']) for item in data['annotations'])
            return points
        else:
            print("The data structure is not as expected.")
            return None
  
    
    def sample_points_from_mesh_within_distance(mesh, soma, max_distance=80000, num_samples=1000):
          """
          Randomly sample points on the surface of a mesh within a specified distance from the soma.
    
          Parameters:
          - mesh (trimesh.Trimesh): The mesh from which to sample points.
          - soma (np.array): The scaled coordinates of the soma position.
          - max_distance (float): Maximum allowed distance from the soma.
          - num_samples (int): Number of points to sample.
    
          Returns:
          - np.array: Array of points sampled from the mesh surface within the specified distance.
          """
          points_within_distance = []
    
          # Keep sampling until we reach the required number of valid points
          while len(points_within_distance) < num_samples:
              remaining_samples = num_samples - len(points_within_distance)
    
              # Calculate the areas of each face of the mesh
              face_areas = mesh.area_faces
    
              # Normalize the areas to a probability distribution
              probabilities = face_areas / face_areas.sum()
    
              # Choose faces randomly according to their area
              chosen_faces = np.random.choice(len(face_areas), size=remaining_samples, p=probabilities)
    
              # Barycentric coordinates for random points within each face
              u = np.random.rand(remaining_samples)
              v = np.random.rand(remaining_samples)
              w = 1 - u - v
              mask = u + v > 1
              u[mask], v[mask] = 1 - u[mask], 1 - v[mask]  # Reflect points back to the triangle if they fall outside
    
              # Calculate the actual coordinates of the points
              vertices = mesh.vertices[mesh.faces[chosen_faces]]
              sampled_points = (vertices[:, 0, :] * u[:, np.newaxis] +
                                vertices[:, 1, :] * v[:, np.newaxis] +
                                vertices[:, 2, :] * w[:, np.newaxis])
    
              # Calculate distances to the soma
              distances_to_soma = np.linalg.norm(sampled_points - soma, axis=1)
    
              # Filter points within the max_distance
              valid_points = sampled_points[distances_to_soma <= max_distance]
    
              # Add valid points to the list
              points_within_distance.extend(valid_points)
    
          # Convert list to numpy array and return exactly num_samples
          return np.array(points_within_distance[:num_samples])

    
  ############################################################################################################################################################  
  
    # set up vars
    opcname = "OPC12" #change n to be any ID you would like
    scaled_soma = np.array([158274, 197919, 16082]) #grab relative center of the soma through neuroglancer  
    soma = scaled_soma * scale
    OPC = 864691135432858866  #change to any OPC segment id
    OPC_mesh = mm.mesh(seg_id=OPC)
    microglia_meshes = np.array([864691135501540674]) #add any microglia seg_id
    contacts = np.empty(len(microglia_meshes),dtype=object)
    num_microglia = len(microglia_meshes)
    PLs_file_path = 'D:\MicronsEMdata\OPC_PLs\OPC12_PLs.json' #Import extracted .json annotations 
        
    #To extract JSON data, click the {} button on the top right corner of the neuroglancer window and copy your annotation layer data into a text editor. Save as a .JSON and should work!
    
    PLs = load_PLs(PLs_file_path)
    
    # Making voxels for phagolysosomes 
    PL_scale =PLs*scale
    PL_scale = PL_scale.reshape(-1,3)
    PL_unscale = PL_scale/scale
    PL_tree = cKDTree(PL_scale)  # Create a KD-tree with scaled PL positions
    dis = soma - PL_scale
    PL_distance_to_soma = np.linalg.norm(dis, axis=1)
    
    PL_data = pd.DataFrame({
        'X': PL_unscale[:, 0],
        'Y': PL_unscale[:, 1],
        'Z': PL_unscale[:, 2],
        'Scaled_X': PL_scale[:, 0],
        'Scaled_Y': PL_scale[:, 1],
        'Scaled_Z': PL_scale[:, 2],
        'Distance_From_Soma': PL_distance_to_soma
    })
    
    # Save PL data to CSV
    PL_filename = Path(f"{base_directory}/{opcname}_Micro_Contacts/{opcname}_PLs.csv")
    PL_filename.parent.mkdir(parents=True, exist_ok=True)
    PL_data.to_csv(PL_filename, index=False)

    # Setting up color maps
    colormap = plt.get_cmap("viridis")
    colors = [colormap(i / num_microglia)[:3] for i in range(num_microglia)] 
    
    # Grabbing random points from the OPC mesh for null hypothesis testing
    num_samples = 1000  # Number of points you want to sample
    sampled_points = sample_points_from_mesh_within_distance(OPC_mesh, soma, max_distance=80000, num_samples=num_samples)
    RanPt_distance_to_soma = np.linalg.norm(sampled_points - soma, axis=1)
    RanPt_dis = RanPt_distance_to_soma.reshape(-1, 1)
    data_pts = np.hstack((sampled_points, sampled_points / scale, RanPt_dis))
    RanOPC_points_filename = Path(f"{base_directory}/{opcname}_Micro_Contacts/{opcname}_RanPoints.csv")
    RanOPC_points_filename.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    np.savetxt(RanOPC_points_filename, data_pts, delimiter=",", header="x,y,z,scaled x, scaled y, scaled z, distance", comments='', fmt='%.6f')
    
    print(f"Sampled points saved to {RanOPC_points_filename}")

########################################################################################################################################################
#The following code is to find all points in which two mesh objects (OPC and microglia) are colliding (touching)  
   
    for i, microglia_id in enumerate(microglia_meshes):
        microglia_mesh = mm.mesh(seg_id=microglia_id)
        CM = collision.CollisionManager()
        CM.add_object('OPC', OPC_mesh)
        CM.add_object('micro', microglia_mesh)
    
        is_collision, contacts = CM.in_collision_internal(return_data=True)
        voxels = defaultdict(list)
    
        for contact in contacts:
            point = contact.point
            voxel_index = tuple((point // voxel_size).astype(int))
            voxels[voxel_index].append(point)
        
        # Deduplicate points within each voxel before further processing
        for voxel_index, points in voxels.items():
            voxels[voxel_index] = unique_points(points)
            
        # Process voxel merging
        merged_voxels = merge_voxels(voxels, voxel_size)
    
    # After merging voxels and before saving data
        merged_output_data = []
        for key, data in merged_voxels.items():
            p = data['points']
            surface_area = calculate_surface_area(p)
            centroid = data['centroid']
            scaled_centroid = centroid / scale
            cont_distance_to_soma = np.linalg.norm(soma - centroid)
            
            # Find the nearest PL and calculate the scaled distance
            _, idx = PL_tree.query(centroid)  # Find nearest index in PL_scale
            nearest_PL_distance = np.linalg.norm(centroid - PL_scale[idx])

            data_entry = [
                key[0], key[1], key[2], len(data['points']),
                *centroid.tolist(), *scaled_centroid.tolist(),
                cont_distance_to_soma, surface_area, nearest_PL_distance
            ]
            merged_output_data.append(data_entry)
        
    
            # Prepare data for CSV output of individual points
            pts = np.array([contact.point for contact in contacts])
            microglia_points[microglia_id] = pts
            if len(pts) == 0:
                print("No collision points detected")
                pts_scaled = "No collisions"
            elif len(pts) > 0:
                pts_scaled = pts/scale
        
            # Save merged voxel data
            merged_filename = Path(f"{base_directory}/{opcname}_Micro_Contacts/{opcname}_micro_{microglia_id}_merged.csv")
            merged_filename.parent.mkdir(parents=True, exist_ok=True)
            merged_header = "Voxel X, Voxel Y, Voxel Z, Number of Points, Centroid X, Centroid Y, Centroid Z, Scaled Centroid X, Scaled Centroid Y, Scaled Centroid Z, Euclidian distance to soma, Surface Area, Nearest PL Distance, PL Distance to Soma"
            np.savetxt(merged_filename, np.array(merged_output_data), delimiter=",", header=merged_header, comments='', fmt='%s')
        
            points_output_data = np.hstack((pts,pts_scaled))
            points_filename = Path(f"{base_directory}/{opcname}_Micro_Contacts/{opcname}_micro_{microglia_id}_points.csv")
            points_header = "Unscaled X Position, Unscaled Y Position, Unscaled Z Position, Scaled X Position, Scaled Y Position, Scaled Z Position"
            np.savetxt(points_filename, points_output_data, delimiter=",", header=points_header, comments='', fmt='%s')
        
            print(f"Saved merged voxel data for {opcname}, microglia {microglia_id} in {merged_filename}")
            print(f"Saved individual collision points for {opcname}, microglia {microglia_id} in {points_filename}")
 
    
 
    
 ##########################################################################################################################################################       
 # visualization of data in render space    
 
 
    # Initialize a list to store all actors
    actors = []
    opc_micro_actors = []
    opc_micro_pts_actors = []
    opc_pts_actors = []
    opc_pl_actors = []
    opc_pl_pts_actors = []
    opc_ranPts_actors = []
    
    # Process each microglia mesh to create and store mesh actors
    for i, microglia_id in enumerate(microglia_meshes):
        microglia_mesh = mm.mesh(seg_id=microglia_id)
        microglia_actor = trimesh_vtk.mesh_actor(microglia_mesh, color=colors[i % len(colors)], opacity=0.2)
        actors.append(microglia_actor)  # Add each microglia actor to the list
        opc_micro_actors.append(microglia_actor)
        opc_micro_pts_actors.append(microglia_actor)
    
    for i, microglia_id in enumerate(microglia_meshes):
        if microglia_id in microglia_points:
            points = microglia_points[microglia_id]  # Get the points for this microglia
            # Ensure points are in the correct format, e.g., an Nx3 numpy array
            if points.size > 0:  # Check if there are points to visualize
                collision_actor = trimesh_vtk.point_cloud_actor(points, color=colors[i % len(colors)], opacity=1)
                actors.append(collision_actor)
                opc_micro_pts_actors.append(collision_actor)
                opc_pts_actors.append(collision_actor)
                opc_pl_pts_actors.append(collision_actor)
    
    # Add OPC mesh actor
    opc_actor = trimesh_vtk.mesh_actor(OPC_mesh, color=(1, 0, 1), opacity=0.1)
    actors.append(opc_actor)
    opc_micro_actors.append(opc_actor)
    opc_micro_pts_actors.append(opc_actor)
    opc_pts_actors.append(opc_actor)
    opc_pl_actors.append(opc_actor)
    opc_pl_pts_actors.append(opc_actor)
    opc_ranPts_actors.append(opc_actor)
    
    # Add sphere actors for phagolysosomes
    PL_actors = create_sphere_meshes(PL_scale, 500, (1, 0, 1))
    RanPts_actors = create_sphere_meshes(sampled_points, 500, (1,1,0))
    actors.extend(PL_actors)  # Use extend to add all elements of PL_actors to the actors list
    opc_pl_actors.extend(PL_actors)
    opc_pl_pts_actors.extend(PL_actors)
    opc_ranPts_actors.extend(RanPts_actors)
    
    #initialize camera for video making
    camera_1 = trimesh_vtk.oriented_camera(OPC_mesh.centroid, backoff=250, backoff_vector=[0, 0, -1], up_vector = [0, -1, 0]) # make up negative y)# center the output vizualzation 200 units away from the centroid of the OPC, 
    trimesh_vtk.render_actors([opc_actor],camera = camera_1)

    frames = 100

    #Make video of just the OPC
    trimesh_vtk.render_actors_360([opc_actor], 
                                 directory = Path(f"{base_directory}/{opcname}_Micro_Contacts/{opcname}/"), 
                                 nframes = frames, 
                                 camera_start = camera_1,
                                 do_save = True)
    
    #Make video of just the OPC with PLs
    trimesh_vtk.render_actors_360(opc_ranPts_actors, 
                                 directory = Path(f"{base_directory}/{opcname}_RanPoints_/{opcname}_RanPts/"), 
                                 nframes = frames, 
                                 camera_start = camera_1,
                                 do_save = True)

    trimesh_vtk.render_actors_360(opc_pl_actors, 
                                 directory = Path(f"{base_directory}/{opcname}_Micro_Contacts/{opcname}_PL/"), 
                                 nframes = frames, 
                                 camera_start = camera_1,
                                 do_save = True)


    #Make video of OPC with Microglia
    trimesh_vtk.render_actors_360(opc_micro_actors, 
                                 directory = Path(f"{base_directory}/{opcname}_Micro_Contacts/{opcname}_micro/"),
                                 nframes = frames,
                                 camera_start = camera_1,
                                 do_save = True)

    #Make video of OPC with collisions
    trimesh_vtk.render_actors_360(opc_pts_actors, 
                                 directory = Path(f"{base_directory}/{opcname}_Micro_Contacts/{opcname}_Pts/"),
                                 nframes = frames,
                                 camera_start = camera_1,
                                 do_save = True)

    #Make video of OPC with collisions and PLs
    trimesh_vtk.render_actors_360(opc_pl_pts_actors, 
                                 directory = Path(f"{base_directory}/{opcname}_Micro_Contacts/{opcname}_Pts_PL/"),
                                 nframes = frames,
                                 camera_start = camera_1,
                                 do_save = True)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
