"""Script to prepare data. Mostly to create more suitable version of dataset with params from settings."""
import pandas as pd
import os 
import logging
import sys
import json
import time
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)

from utils import singleton, get_project_dir, configure_logging, extract_zip

#configure and creating logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CONF_FILE = "settings.json"

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(os.path.join(SRC_DIR, CONF_FILE), "r") as file:
    conf = json.load(file)

processed_data_dir = conf['processing']['processed_data_dir']
kaggle_data_dir = os.path.join(ROOT_DIR, conf['general']['kaggle_data_dir'])
    
if not os.path.exists(kaggle_data_dir):
    os.makedirs(kaggle_data_dir)

if not os.path.exists(processed_data_dir):
    os.path.makedirs(processed_data_dir)

class DataProcessor():
    def __init__(self, data_path:os.path, size_threshold, empty_amount, has_ships_amount) -> None:
        self.input_df = pd.read_csv(data_path)
        self.processing_df = self.input_df
        self.size_threshold = size_threshold
        self.empty_amount = empty_amount
        self.has_ships_amount = has_ships_amount

    def get_ships_feature(self):
        """function to get column get has_ship"""
        self.processing_df['has_ship'] = self.processing_df['EncodedPixels'].map(lambda x: 1 if x is not np.NaN else 0)

    def aggregate_df(self):
        """function to aggregate df"""
        self.processing_df = self.processing_df.groupby('ImageId').agg({'has_ship':'sum'}).reset_index()
    
    def remove_undersize_ids(self):
        """function to remove ids, with image_size <= threshold"""
        self.processing_df['file_size_kb'] = self.processing_df['ImageId'].map(lambda c_img_id: 
                                                               os.stat(os.path.join(os.path.join(kaggle_data_dir, conf['general']['train_images_folder_name']), 
                                                                                    c_img_id)).st_size/1024)
        self.processing_df = self.processing_df[self.processing_df['file_size_kb'] > self.size_threshold]
    
    def sample_df(self):
        """function to balance df with images without ships and with"""
        self.processing_df = pd.concat([self.input_df[self.input_df["EncodedPixels"].isna()].sample(self.empty_amount),
                         self.input_df[~self.input_df["EncodedPixels"].isna()].sample(self.has_ships_amount)])
    
    def save_df(self, path_to_save ):
        """function to save df"""
        self.processing_df.to_csv(path_to_save)
    
    def run_processing_pipeline(self, path):
        """function to run pipeline"""
        start_time = time.time()
        logger.info("Starting preprocessing pipeline...")
        logger.info("Getting ships feature")
        self.get_ships_feature()
        logger.info("Aggregating df")
        self.aggregate_df()
        logger.info("Removing under size_threshold image ids ")
        self.remove_undersize_ids()
        logger.info(f"Sampling df with {self.empty_amount} images without ship, and {self.has_ships_amount} images with ship..")
        self.sample_df()
        logger.info(f"Saving preprocessed df to {path}")
        self.save_df(path)
        end_time = time.time()
        logger.info(f"Preprocessing pipeline finished in {end_time - start_time} s.")
        
def main():
    configure_logging()
    dataProcessor = DataProcessor(os.path.join(kaggle_data_dir, conf['processing']['input_df_name']), 
                                  size_threshold=conf['processing']['size_threshold'],
                                  empty_amount=conf['processing']['empty_amount'],
                                  has_ships_amount=conf['processing']['has_ships_amount'])
    
    dataProcessor.run_processing_pipeline(os.path.join(processed_data_dir, conf['train']['table_name']))


if __name__ == "__main__":
    main()