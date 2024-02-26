"""Script to upload data for training and testing"""

# Importing required libraries
import argparse
import os 
import logging
import sys
import json
import pandas as pd
import time
import zipfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)

from utils import singleton, get_project_dir, configure_logging

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

os.environ['KAGGLE_USERNAME']=kaggle_api["username"]
os.environ['KAGGLE_KEY']=kaggle_api["key"]

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

parser = argparse.ArgumentParser()
parser.add_argument("--mode",
                    help="Specify data to load (prepared - small version, like in notebook, full - full data, inference - test data from kaggle) prepared/full/inference",
                    default="prepared")

if __name__ == '__main__':
    configure_logging()
    logger.info("Starting DataLoading script...")
    args = parser.parse_args()
    dataset_name = conf['processing']['dataset_name']
    api = KaggleApi()
    api.authenticate()
    train_images_dir_path = os.path.join(ROOT_DIR, conf['general']['train_images_dir'])
    downloaded_csvs_path = os.path.join(ROOT_DIR, conf['processing']['downloaded_csv_dir'])
    test_images_dir_path = os.path.join(ROOT_DIR, conf['general']['test_images_dir'])
    if not os.path.exists(train_images_dir_path):
        os.makedirs(train_images_dir_path)
    if not os.path.exists(downloaded_csvs_path):
        os.makedirs(downloaded_csvs_path)
    if not os.path.exists(test_images_dir_path):
            os.makedirs(test_images_dir_path)

    if args.mode == 'prepared':
        ids_df = pd.read_csv(os.path.join(ROOT_DIR, 'data/prepared/prep_data.csv'))
        logger.info("Starting downloading prepared data...")
        start_time = time.time()
        for id in ids_df['ImageId']:
            try:
                kaggle.api.competition_download_file("airbus-ship-detection", 
                                                     f"train_v2/{id}", 
                                                     path=train_images_dir_path,
                                                       quiet=True)
            except Exception as e:
                logger.info(f"Failed to download file with ID {id}: {str(e)} ")
        end_time = time.time()
        logger.info(f"Downloading finished in {end_time - start_time} s")

    if args.mode == 'full':
        logger.info("Starting downloading full train data...")
        start_time = time.time()
        try:
            kaggle.api.competition_download_file("airbus-ship-detection", 
                                                    "train_ship_segmentations_v2.csv", 
                                                    path=downloaded_csvs_path,
                                                    quiet=True)
        except Exception as e:
            logger.info(f"Failed to download train_v2.csv file")

        logger.info("Csv data downloaded, starting extracting.")
        with zipfile.ZipFile(os.path.join(downloaded_csvs_path, 'train_ship_segmentations_v2.csv.zip'), 'r') as zip_ref:
            zip_ref.extractall(downloaded_csvs_path)
        logger.info("Extraction finished, starting downloading images...")
        train_df = pd.read_csv(os.path.join(downloaded_csvs_path, "train_ship_segmentations_v2.csv"))
        for id in train_df['ImageId']:
            try:
                kaggle.api.competition_download_file("airbus-ship-detection", 
                                                     f"train_v2/{id}", 
                                                     path=train_images_dir_path,
                                                       quiet=True)
            except Exception as e:
                logger.info(f"Failed to download file with ID {id}: {str(e)} ")
        end_time = time.time()
        logger.info(f"Downloading full train data finished in {end_time - start_time} s")
    if args.mode == "inference":
        logger.info("Starting downloading inference data...")
        start_time = time.time()
        try:
            kaggle.api.competition_download_file("airbus-ship-detection", 
                                                    "sample_submission_v2.csv", 
                                                    path=downloaded_csvs_path,
                                                    quiet=True)
        except Exception as e:
            logger.info(f"Failed to download sample_submission.csv file")

        logger.info("Csv data downloaded, starting downloading images...")
        inference_df = pd.read_csv(os.path.join(downloaded_csvs_path, "sample_submission_v2.csv"))
        for id in inference_df['ImageId'][:10]:
            try:
                kaggle.api.competition_download_file("airbus-ship-detection", 
                                                     f"test_v2/{id}", 
                                                     path=test_images_dir_path,
                                                       quiet=True)
            except Exception as e:
                logger.info(f"Failed to download file with ID {id}: {str(e)} ")
        end_time = time.time()
        logger.info(f"Downloading inference data finished in {end_time - start_time} s")
        
        
        

