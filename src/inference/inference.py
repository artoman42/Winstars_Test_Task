"""
This script uses trained model to make predictions.
"""

import argparse
import os
import sys
import json
import logging
import pandas as pd
import time
from skimage.io import imread
import numpy as np

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
MODELS_DIR = os.path.join(SRC_DIR, 'models')

sys.path.append(SRC_DIR)
sys.path.append(MODELS_DIR)

from base_unet import Unet_model
from utils import configure_logging, init_losses
from image_utils import get_run_length_encoded_predictions
#configure and creating logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CONF_FILE = "settings.json"

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(os.path.join(SRC_DIR, CONF_FILE), "r") as file:
    conf = json.load(file)

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", 
                    help="Specify data dir", 
                    default=os.path.join(ROOT_DIR, conf['general']['kaggle_data_dir']))

parser.add_argument("--model_weights_path", 
                    help="Specify model weights path", 
                    default=os.path.join(ROOT_DIR, 
                                         conf['inference']['models_weigths_dir'],
                                           conf['inference']['models_weigths_name']))
parser.add_argument("--amount",
                    help="Specify raw amount of test data, on which you wanna test inference",
                    default=1000)
PREDICTIONS_DIR = os.path.join(ROOT_DIR, conf['inference']["predictions_dir"])

if not os.path.exists(PREDICTIONS_DIR):
    os.makedirs(PREDICTIONS_DIR)

class Inference():
    def __init__(self, model, df_path, test_images_path ) -> None:
        self.model = model
        self.df = pd.read_csv(df_path)
        self.test_images_path = test_images_path
        self.results = None

    def init_model(self, model_weigths_path, compile_args):
        """function to init model with trained weights"""
        logger.info(f"Initializing model with weigths from {model_weigths_path}...")
        self.model.compile(**compile_args)
        self.model.load_weights(model_weigths_path)

    def predict(self, img_name, IMG_SCALING=(3,3)):
        """function to predict image mask, it rescales inputs image with IMG_SCALING"""
        c_path = os.path.join( self.test_images_path, img_name)
        c_img = imread(c_path)
        img = np.expand_dims(c_img, 0)/255.0
        if IMG_SCALING is not None:
            img = img[:, ::IMG_SCALING[0], ::IMG_SCALING[1]]
        return img, self.model.predict(img, verbose=0)
    
    def predict_and_encode(self, test_img_names,amount):
        """function to predict and encode"""
        list_dict = []
        logger.info("Making predictions and encodings")
        for img_name in test_img_names[:amount]:
            _ , pred = self.predict(img_name)
            rle_pred = get_run_length_encoded_predictions(pred[0], img_name)
            list_dict += rle_pred
        return pd.DataFrame(list_dict, columns=["ImageId", "EncodedPixels"])   
        
    def save_predictions(self, path_to_save):
        """function to save prediction"""
        logger.info(f"Saving predictions to {path_to_save}")
        self.results.to_csv(path_to_save)

    def run_inference(self, model_weigths_path, compile_args, path_to_save, amount):
        """function to run inference pipeline"""
        logger.info("Starting inference pipeline...")
        start_time = time.time()
        init_losses()
        self.init_model(model_weigths_path, compile_args)
        self.results = self.predict_and_encode(self.df['ImageId'].values, amount)
        self.save_predictions(path_to_save)
        end_time = time.time()
        logger.info(f"Inference finished in {end_time - start_time} s")

if __name__ == "__main__":
    configure_logging()
    # Defines paths
    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        logger.error(f"Data dir - {args.data_path} doesnt't exist ")
    else:
        DATA_DIR = args.data_path

    if not os.path.exists(args.model_weights_path):
        logger.error(f"Data dir - {args.model_weights_path} doesnt't exists ")
    else:
        MODEL_WEIGHTS_PATH = args.model_weights_path
    
    try:
        amount = int(args.amount)
    except:
        logger.error(f"Amount must be integer!")
    

    TEST_PATH = os.path.join(DATA_DIR, conf['inference']['table_name'])
    TEST_IMGAGES_PATH = os.path.join(DATA_DIR, conf['general']['test_images_folder_name'])
    if not os.path.exists(TEST_PATH):
        logger.error(f"{TEST_PATH} doesn't exist ")

    if not os.path.exists(TEST_IMGAGES_PATH):
        logger.error(f"{TEST_IMGAGES_PATH} doesn't exist")
    
    inference = Inference(Unet_model().build_architecture(), TEST_PATH, TEST_IMGAGES_PATH)
    inference.run_inference(MODEL_WEIGHTS_PATH, conf['train']['model_compile'],
                            os.path.join(PREDICTIONS_DIR, conf['inference']['predictions_name']),
                            amount)
    
    


    