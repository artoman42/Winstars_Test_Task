"""DataGenerator class for training, testing and inference process"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)

# from utils import configure_logging
from image_utils import crop3x3_mask, crop3x3, rle_decode

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, datapath, batch_size, df_mask: pd.DataFrame, augmentation_dict=None, cropped_image_size=256):
        self.cropped_image_size = cropped_image_size
        self.datapath = datapath
        self.batch_size = batch_size
        self.df = df_mask.sample(frac=1)
        self.l = len(self.df) // batch_size
        self.augmentation = None
        if augmentation_dict is not None:
            self.augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
                **augmentation_dict
            )

    def __len__(self):
        return self.l

    def on_epoch_end(self):
        self.df = self.df.sample(frac=1)

    def __getitem__(self, index):
        mask = np.empty((self.batch_size, self.cropped_image_size, self.cropped_image_size), np.float32)
        image = np.empty((self.batch_size, self.cropped_image_size, self.cropped_image_size, 3), np.float32)
        batch_df = self.df[index * self.batch_size: (index + 1) * self.batch_size]
        for b, _, row in zip(range(self.batch_size), range(len(batch_df)), batch_df.itertuples()):
            temp = tf.keras.preprocessing.image.load_img(self.datapath + '/' + row.ImageId)
            temp = tf.keras.preprocessing.image.img_to_array(temp) / 255
            mask[b], i = crop3x3_mask(
                rle_decode(
                    row.EncodedPixels
                )
            )
            image[b] = crop3x3(temp, i)

        if self.augmentation is not None:
            augmented_images = []
            augmented_masks = []
            for i in range(self.batch_size):
                augmented = self.augmentation.flow(np.expand_dims(image[i], axis=0),
                                                    np.expand_dims(mask[i], axis=0),
                                                    batch_size=1)
                augmented_image, augmented_mask = next(augmented)
                augmented_images.append(augmented_image.squeeze())
                augmented_masks.append(augmented_mask.squeeze())
            image = np.array(augmented_images)
            mask = np.array(augmented_masks)
            return image, mask
        
    def show_samples(self, num_samples=5):
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
        indices = np.random.randint(0, len(self.df), num_samples)
        
        for i, idx in enumerate(indices):
            row = self.df.iloc[idx]
            image_path = os.path.join(self.datapath, row['ImageId'])
            temp = tf.keras.preprocessing.image.load_img(image_path)
            temp = tf.keras.preprocessing.image.img_to_array(temp) / 255
            mask, _ = crop3x3_mask(rle_decode(row['EncodedPixels']))
            image = crop3x3(temp, _)
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f"Image {i+1}")
            axes[i, 0].axis('off')
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title(f"Mask {i+1}")
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.show()