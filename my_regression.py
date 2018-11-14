import sys
import os
import glob
import numpy as np
import pandas as pd
import keras.backend as kb
from tqdm import tqdm
from imgaug import augmenters as iaa
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.callbacks import Callback
from keras.optimizers import Adam, SGD
from keras.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score

from generator import AugmentedImageSequence
from model_zoo import ModelFactory
import tool


class config(object):
    """ Regression model configuration """

    CLASS_NAMES = ["bone_age"]
    ADD_FEATURES = None
    MODEL_NAME = "InceptionV3"
    EPOCH = 200
    BATCH_SIZE = 16
    NET_INPUT_DIM = 500
    LEARNING_RATE = 1e-3
    
    LOSS = "mae"
    METRICS = ["mae", "mse"]
    GENERATOR_WORKERS = 8
    
    TRAIN_STEPS = 630
    VAL_STEPS = 78

    # Either (a) Directory of images (b) CSV filepath
    #########################################################
    TRAIN = "C:/Users/hliu/Desktop/Boneage/train.csv"
    VAL = "C:/Users/hliu/Desktop/Boneage/val.csv"
    #########################################################


def augmentation():
    """ Real-time image augmentation """
    return iaa.Sequential(
                [    
                iaa.Fliplr(0.5),
                # iaa.Flipud(0.5),
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, 
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, 
                        rotate=(-20, 20),)
                # iaa.CropAndPad(percent=(-0.2,0.2)),
                # iaa.Add((-10, 10), per_channel=0.5),
                # iaa.ContrastNormalization((0.8,1.4),per_channel=0.5),
                # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.85, 1.25)),
                # iaa.Multiply((0.9, 1.1), per_channel=0.5),
                ])



class MyRegression(object):
    def __init__(self):
        self.model = None

    def train(self, log_dir, show_model=True):
        """ Training Regression model
        """
        ###################################################################################
        augs = augmentation()
        optimizer = Adam(lr=config.LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=20, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1,
                                      mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
        ###################################################################################

        TRAIN_CSV_FP, tmp_train = tool.prepare_dataset(config.TRAIN)
        VAL_CSV_FP,   tmp_val   = tool.prepare_dataset(config.VAL)

        SAVE_WEIGHT_FP = os.path.join(log_dir, "{epoch:03d}-{val_loss:.4f}.h5")

        # Make log directory if not exist
        if not os.path.isdir(log_dir): 
            os.makedirs(log_dir)

        # Training dataset
        train_sequence = AugmentedImageSequence(
            csv_fp=TRAIN_CSV_FP,
            class_names=config.CLASS_NAMES,
            batch_size=config.BATCH_SIZE,
            target_size=(config.NET_INPUT_DIM, config.NET_INPUT_DIM),
            steps=config.TRAIN_STEPS,
            additional_features=config.ADD_FEATURES,
            augmenter=augs,
            )

        # Validation dataset
        validation_sequence = AugmentedImageSequence(
            csv_fp=VAL_CSV_FP,
            class_names=config.CLASS_NAMES,
            batch_size=config.BATCH_SIZE,
            target_size=(config.NET_INPUT_DIM, config.NET_INPUT_DIM),
            steps=config.VAL_STEPS,
            additional_features=config.ADD_FEATURES,
            shuffle_on_epoch_end=False,
            )

        # Load Regression model
        model = ModelFactory().get_regression_model(class_num=len(config.CLASS_NAMES),
                                                    model_name=config.MODEL_NAME,
                                                    base_weights=None,
                                                    input_shape=(config.NET_INPUT_DIM,config.NET_INPUT_DIM,3))
        if show_model: print(model.summary())

        model.compile(optimizer=optimizer, loss=config.LOSS, metrics=config.METRICS)

        # Callbacks
        checkpoint = ModelCheckpoint(SAVE_WEIGHT_FP, save_weights_only=False, save_best_only=False, verbose=0)
        tensorboard = TensorBoard(log_dir=os.path.join(log_dir, "logs"))
        csv_logger = CSVLogger(os.path.join(log_dir, "my_logger.csv"))
        callbacks = [checkpoint, tensorboard, csv_logger, early_stop, reduce_lr]

        history = model.fit_generator(
                                    generator=train_sequence,
                                    steps_per_epoch=config.TRAIN_STEPS,
                                    epochs=config.EPOCH,
                                    verbose=2,
                                    validation_data=validation_sequence,
                                    callbacks=callbacks,
                                    workers=config.GENERATOR_WORKERS,
                                    shuffle=False
                                    )

        print("\nFinished training")
        if tmp_train: os.remove(TRAIN_CSV_FP)
        if tmp_val: os.remove(VAL_CSV_FP)


    def load_model(self, model_fp):
        self.model = load_model(model_fp)
        print(f"successfully loaded model: {model_fp}")


    def predict(self, image_fp):  
        '''
        Returns:
        Predicted y.
        '''    
        assert self.model is not None, "Please load model"
        image = tool.read_image(image_fp,3)
        image = tool.resize_image(image,(config.NET_INPUT_DIM,config.NET_INPUT_DIM))
        image = tool.normalize(image)
        batch = np.asarray([image])
        y = np.squeeze(self.model.predict(batch),0).tolist()
        return y


    def write_rsult(self, test_src, save_fp, threshold=0.5):
        """ Write regression results as a file
        test_src: either (a) test_dir (b) test_csv_fp
        """
        assert self.model is not None, "Please load model"
        class_names = config.CLASS_NAMES
        row =",".join(x for x in ["image_fp"]+class_names)+"\n"
        image_fps = None
        if os.path.isdir(test_src): image_fps = glob.glob(image_dir+"/*.*")
        if os.path.isfile(test_src):
            df_test = pd.read_csv(test_src)
            pid_column_name = df_test.columns.tolist()[0]
            image_fps = df_test[pid_column_name].tolist()
        with open(save_fp, 'w') as f:
            f.write(row)
            for image_fp in tqdm(image_fps):
                y = self.predict(image_fp)
                target = [round(x, 3) for x in y]
                target_str = ",".join(str(x) for x in target)
                f.write(image_fp+","+target_str+"\n")
