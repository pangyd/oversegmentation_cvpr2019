#!/bin/bash

# Download HDF5 for indoor 3d semantic segmentation (around 1.6GB)
wget -O D://data/point_cloud/indoor3d_sem_seg_hdf5_data.zip https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip
unzip indoor3d_sem_seg_hdf5_data.zip
rm indoor3d_sem_seg_hdf5_data.zip

