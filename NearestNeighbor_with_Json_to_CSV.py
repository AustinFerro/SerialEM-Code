# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:31:05 2024

@author: Austin Ferro 

Data that will take .JSON data created from other scripts and find nearest neighbor distances and convert to CSV 

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


base_directory = f"C:/Users/thech/Documents/PyCode/MicronsEMdata" #change to whatever base directory you would like to save data to

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

def main():
    # set up vars
    opcname = "OPC16" #change n to be any ID you would like
    scaled_soma = np.array([250118, 159369, 23649]) #grab relative center of the soma through neuroglancer  
    soma = scaled_soma * scale


    Syn_file_path = f"F:/MicronsEMdata/OPC_Synapses/{opcname}_Synapses.json"
    PLs_file_path = f"F:/MicronsEMdata/OPC_PLs/{opcname}_PLs.json"
    Contact_file_path = f"F:/MicronsEMdata/OPC_ContactPoints/{opcname}_ContactPoints.json"

    syn_points = load_PLs(Syn_file_path) #not scaled
    pls_points = load_PLs(PLs_file_path)
    contact_points = load_PLs(Contact_file_path)
    contact_points = contact_points[1:]

    contact_points_array = np.array([tuple(map(float, coords)) for coords in contact_points])    
    
    syn = syn_points *scale 
    contact = contact_points_array*scale
    pls = pls_points*scale

    syn = syn.reshape(-1,3)
    syn_points = syn/scale
    
    pls = pls.reshape(-1,3)
    pls_scale = pls/scale
    
    syn_tree = cKDTree(syn)
    
    dis = soma - syn
    syn_distance_to_soma = np.linalg.norm(dis, axis=1)
    
    dis_pls = soma - pls
    pls_distance_to_soma = np.linalg.norm(dis_pls, axis=1)

    
    
    for i, contacts in enumerate(contact):
        # Query the nearest distance and index for this contact point
        nnsyn_contact_distance, index = syn_tree.query(contact)
    
    syn_df = pd.DataFrame({
        'X': syn_points[:, 0],
        'Y': syn_points[:, 1],
        'Z': syn_points[:, 2],
        'Scaled_X': syn[:, 0],
        'Scaled_Y': syn[:, 1],
        'Scaled_Z': syn[:, 2],
        'Synapse Distance to Soma': syn_distance_to_soma
    })
    
    nnFilename = f"F:/MicronsEMdata/OPC_Synapses/{opcname}_syn_contact_nearestNeighbor_distance.csv"
    np.savetxt(nnFilename, np.array(nnsyn_contact_distance), delimiter="", comments='', fmt='%s')
   
    # Save syn data to CSV
    synPos_fileName = f"F:/MicronsEMdata/OPC_Synapses/{opcname}_syn_Pos.csv"
    syn_df.to_csv(synPos_fileName, index=False) 
   
    pl_df = pd.DataFrame({
        'X': pls_scale[:, 0],
        'Y': pls_scale[:, 1],
        'Z': pls_scale[:, 2],
        'Scaled_X': pls[:, 0],
        'Scaled_Y': pls[:, 1],
        'Scaled_Z': pls[:, 2],
        'Synapse Distance to Soma': pls_distance_to_soma
    }) 
   
    

    for i, PLs in enumerate(pls):
        # Query the nearest distance and index for this contact point
        nnPL_syn_distance, index = syn_tree.query(pls)

    nnFilename2 = f"F:/MicronsEMdata/OPC_PLs/{opcname}_syn_PL_nearestNeighbor_distance.csv"
    np.savetxt(nnFilename2, np.array(nnPL_syn_distance), delimiter="", comments='', fmt='%s')
    
    # Save syn data to CSV
    PlPos_fileName = f"F:/MicronsEMdata/OPC_PLs/{opcname}_PL_Pos.csv"
    pl_df.to_csv(PlPos_fileName, index=False) 

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

    







