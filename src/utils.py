"""This script includes different helper functions."""

import sys
import os
import logging
import zipfile
import tensorflow as tf
from losses import dice_score, BCE_dice
from tensorflow.keras.utils import get_custom_objects

#add pathes to s
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
MODELS_DIR = os.path.join(SRC_DIR, 'models')
sys.path.append(MODELS_DIR)

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

def get_project_dir(sub_dir: str) -> str:
    """Return path to a project subdirectory."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), sub_dir))

def configure_logging() -> None:
    """Configures logging"""
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
def extract_zip(zip_file_path, extract_to):
    """Extracts zipfile"""
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def try_to_use_gpus():
    """This function tries to enable GPUs for training process."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    logging.info("Trying to find GPUs...")
    if gpus:
        try:
            # Set memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logging.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            logging.error(e)
    else:
        logging.info("No capable GPUs found! Using only CPU!")

def init_losses():
        """function to add my own created losses to scope of tensorflow"""
        get_custom_objects().update({"BCE_dice": BCE_dice,
                              "dice_score":dice_score})
 