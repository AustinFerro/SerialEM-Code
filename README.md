# SerialEM_PythonCodeandMatLabCode
#Austin Ferro 2024

This is a repository of python and matlab scripts and data that were used to analyze the Microns cortical 1mm^3 dataset (https://www.microns-explorer.org/cortical-mm3)

In brief Cell_Colli.py is a script that will take multiple segmented objects within the Microns dataset and localize where those objects collide. Cell_Collie will read in .json files of specific locations (in this analysis, the locations of phagolysosomes(PLs)) to see where the collisions are in relation to those objects as well as provide randomly sampled positions on any specific segmented object's mesh. 

The SimulatedPls.m script will then load in those random sampled locations, as well as other data to run a Monte Carlo simulation to examine whether collisions are closer to your locations of interest (PLs in this case) than a random distribution. 


#Dependencies 

fcl (pip install python-fcl)
from caveclient import CAVEclient #https://github.com/CAVEconnectome/CAVEclient
from meshparty import trimesh_io, trimesh_vtk, skeletonize, skeleton, skeleton_io
import cloudvolume
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
