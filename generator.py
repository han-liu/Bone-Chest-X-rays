import os
import random
import numpy as np
import pandas as pd
from keras.utils import Sequence

import sys
sys.path.append("C:/Users/hliu/Desktop/DL/toolbox")
import tool


class AugmentedImageSequence(Sequence):
    def __init__(self, csv_fp, class_names, batch_size,
            target_size, steps, augmenter=None, 
            shuffle_on_epoch_end=True, random_state=1):

            self.csv_fp = csv_fp
            self.batch_size = batch_size
            self.target_size = target_size
            self.augmenter = augmenter
            self.steps = int(steps)
            self.shuffle = shuffle_on_epoch_end
            self.random_state = random_state
            self.class_names = class_names
            self.prepare_dataset()

    def __bool__(self):
        return True

    def __len__(self):
        max_step_size = int(np.floor(len(self.image_fps)/self.batch_size))
        if self.steps > max_step_size:
            self.steps = max_step_size
        return self.steps

    def __getitem__(self, idx):
        batch_image_fps = self.image_fps[self.batch_size*idx : self.batch_size*(idx + 1)]
        batch_label = self.labels[self.batch_size*idx : self.batch_size*(idx + 1)]
        batch_images = np.asarray([self.load_image(image_fp) for image_fp in batch_image_fps])
        batch_images = self.transform_batch_images(batch_images)
        return batch_images, batch_label

    def load_image(self, image_fp):
        image = tool.read_image(image_fp, 3)
        image = tool.resize_image(image, self.target_size)
        image = tool.normalize(image)
        return image

    def transform_batch_images(self, batch_images):
        if self.augmenter is not None:
            batch_images = self.augmenter.augment_images(batch_images)
        return batch_images

    def get_y_true(self):
        if self.shuffle:
            raise ValueError("'To get y_true, shuffle_on_epoch_end' should be false.")
        return self.labels[:self.steps*self.batch_size, :]

    def prepare_dataset(self):
        # The random seed will be different at each epoch
        df = pd.read_csv(self.csv_fp).sample(frac=1., random_state=self.random_state)
        self.image_fps, self.labels = df.iloc[:,0].as_matrix(), df[self.class_names].as_matrix()

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()