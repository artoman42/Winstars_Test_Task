"""Script to upload data from kaggle. Requires your kaggle api in json format, by path - src/api-keys/kaggle.json"""

# Importing required libraries
import os 
import logging
import sys
import json
import pandas as pd
import time

#define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)

from utils import configure_logging, extract_zip

#configure and creating logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CONF_FILE = "settings.json"

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(os.path.join(SRC_DIR, CONF_FILE), "r") as file:
    conf = json.load(file)

#providing kaggle credentials
with open(os.path.join(ROOT_DIR, conf['general']['kaggle_api_path']), "r") as file:
    kaggle_api = json.load(file)

#loading kaggle creadentials into environment
os.environ['KAGGLE_USERNAME']=kaggle_api["username"]
os.environ['KAGGLE_KEY']=kaggle_api["key"]

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    configure_logging()
    logger.info("Starting DataLoading script...")
    dataset_name = conf['processing']['dataset_name']
    api = KaggleApi()
    api.authenticate()
    
    kaggle_data_dir = os.path.join(ROOT_DIR, conf['general']['kaggle_data_dir'])
    
    if not os.path.exists(kaggle_data_dir):
        os.makedirs(kaggle_data_dir)

    logger.info("Downloading data...")
    start_time = time.time()
    logger.info(kaggle_data_dir)
    kaggle.api.competition_download_files("airbus-ship-detection", path=kaggle_data_dir)
    zip_file_path = os.path.join(kaggle_data_dir, f"{dataset_name.split('/')[1]}.zip")
    logger.info("Data downloaded.")
    logger.info("Starting extracting...")
    logger.info(zip_file_path)
    extract_zip(zip_file_path, kaggle_data_dir)
    os.remove(zip_file_path)
    end_time = time.time()

    logger.info(f"Downloading and extracting data finished in {(end_time - start_time)/60} minutes.")

if __name__ == '__main__':
    main()
    
        

