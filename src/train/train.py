"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import json
import logging
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
DATA_PROCESS_DIR = os.path.join(SRC_DIR, 'data_process')
MODELS_DIR = os.path.join(SRC_DIR, 'models')
sys.path.append(SRC_DIR)
sys.path.append(DATA_PROCESS_DIR)
sys.path.append(MODELS_DIR)


from utils import configure_logging, try_to_use_gpus
from DataGenerator import DataGenerator 
from base_unet import Unet_model
from callbacks import get_callbacks

#configure and creating logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Loads configuration settings from JSON
CONF_FILE = "settings.json"
with open(os.path.join(SRC_DIR, CONF_FILE), "r") as file:
    conf = json.load(file)

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", 
                    help="Specify data dir", 
                    default=os.path.join(ROOT_DIR, conf['general']['kaggle_data_dir']))

class Training():
    def __init__(self, df_path, images_path, model):
        self.model = model
        self.images_path = images_path
        self.df = pd.read_csv(df_path)
        self.train = None
        self.valid = None
        self.history = None
        
    
    def prepare_data(self, images_path):
        """function to prepare image data for training"""
        logger.info("Splitting data to train/validation subsets...")
        train_df, valid_df = train_test_split(self.df, test_size=conf['train']['validation_ratio'])
        self.train = DataGenerator(images_path,
                                    conf['train']['batch_size'],
                                    train_df,
                                      conf['train']['augmentation_dict'])
        self.valid = DataGenerator(images_path,
                                    conf['train']['batch_size'],
                                    valid_df)
        
    def compile_model(self, compile_args=conf['train']['model_compile']):
        """function to compile model"""
        logger.info("Compiling model...")
        self.model.compile(**compile_args)
    
    def run_training(self):
        """function to run training"""
        logger.info("Starting training pipeline")
        start_time = time.time()
        try_to_use_gpus()
        self.init_losses()
        self.prepare_data(self.images_path)
        self.compile_model()
        self.model.fit(self.train,
                       epochs= conf['train']['epochs'],
                       batch_size=conf['train']['batch_size'],
                       validation_data=self.valid,
                       callbacks=get_callbacks())
        end_time = time.time()
        logger.info(f"Training finished in {(end_time - start_time)/60} minutes.")    
    

if __name__ == "__main__":
    configure_logging()
    # Defines paths
    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        logger.error(f"Data dir - {args.data_path} doesnt't exists ")
    else:
        DATA_DIR = args.data_path

    CHECKPOINTS_PATH = os.path.join(ROOT_DIR, conf['callbacks']['checkpoint']['filepath'])
    PREPROCESSED_DATA_DIR = os.path.join(ROOT_DIR, conf['processing']['processed_data_dir'])
    TRAIN_PATH = os.path.join(PREPROCESSED_DATA_DIR, conf['train']['table_name'])
    IMAGES_PATH = os.path.join(DATA_DIR, conf['general']['train_images_folder_name'])

    if not os.path.exists(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH)

    if not os.path.exists(PREPROCESSED_DATA_DIR):
        logger.error(f"Prepocessed data - {PREPROCESSED_DATA_DIR} not found.")

    if not os.path.exists(IMAGES_PATH):
        logger.error(f"{IMAGES_PATH} doesn't exists.")
    
    training = Training(TRAIN_PATH, IMAGES_PATH, Unet_model().build_architecture())

    training.run_training()

    
    
