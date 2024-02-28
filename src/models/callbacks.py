"""Script for creating callbacks"""

import json
import logging 
import os
import sys

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)

from utils import configure_logging

#configure and creating logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CONF_FILE = "settings.json"

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(os.path.join(SRC_DIR, CONF_FILE), "r") as file:
    conf = json.load(file)

def get_callbacks():
    """Function to create and list of callbacks, by params from settings, json"""
    logger.info("Creating callbacks...")

    return [ModelCheckpoint(**conf['callbacks']['checkpoint']),
            ReduceLROnPlateau(**conf['callbacks']['reduceLrOnPLat']),
            EarlyStopping(**conf['callbacks']['early_stopping'])]

if __name__ == '__main__':
    """
    Main function was used just for testing if all callbacks were created succesfully.
    In practice will be used only function get_callbacks.
    """
    configure_logging()
    callbacks = get_callbacks()
    successfully_created_callbacks = []
    for callback in callbacks:
        try:
            # Try to create each callback
            successfully_created_callbacks.append(callback)
            logger.info(f"Callback {type(callback).__name__} was created successfully.")
        except Exception as e:
            logger.error(f"Failed to create callback {type(callback).__name__}: {e}")

    if len(successfully_created_callbacks) == len(callbacks):
        logger.info("All callbacks were created successfully.")
    else:
        logger.warning("Some callbacks were not created successfully.")