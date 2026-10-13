# AIS Fishing Detector

This project provides a pipeline for analyzing and classifying fishing activity in AIS (Automatic Identification System) data for trawlers. It compares traditional machine learning approaches, specifically Random Forest, with deep learning temporal models using Long Short-Term Memory (LSTM) networks.

## Dataset
The project utilizes historical AIS data. The original source of the `trawlers.csv` dataset is the Global Fishing Watch [public training data](https://globalfishingwatch.org/data-download/datasets/public-training-data-v1), which requires login for access.

Note on data format: where possible this project uses Parquet files for intermediate and input datasets. Parquet reduces memory footprint when loading data into Google Colab and significantly reduces download/upload sizes when transferring files to/from Google Drive. Using Parquet is recommended for reproducible Colab experiments and to minimize local memory pressure when working with large AIS datasets.

## Features
- **Feature Engineering**: A complete pipeline for data cleaning, voxel segmentation, and feature extraction (temporal and spatial).
- **Modeling**: 
    - Random Forest Classifier (baseline)
    - Long Short-Term Memory (LSTM) network for temporal behavior learning.
- **Visualization**: Interactive Leaflet maps to visualize fishing behavior and model performance.

## Performance
While the Random Forest model often demonstrates high accuracy on training sets, the LSTM model proves more robust when applied to completely unseen data. The LSTM is particularly effective at classifying seemingly haphazard vessel movements as fishing activity in the generated visualization maps.

## Directory Structure
Please be advised that this project was originally developed in Google Colab. The codebase assumes a specific environment structure. Viewers should exercise caution regarding folder paths and directory structures, as they may need to be adjusted based on your local environment setup.

The current structure is:
- `AIS_Dataset/`: Input parquet files.
- `AIS_Label/`: Label parquet files.
- `AIS_Output/`: Generated interactive maps.
- `*.ipynb`: Jupyter notebooks for data processing, training, and evaluation.
- `*.py`: Supporting Python scripts for map visualization and utility.

## Prerequisites
- Python 3.10+
- Dependencies listed in `requirements.txt`
