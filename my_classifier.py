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
from sklearn.metrics import roc_auc_score

from generator import AugmentedImageSequence
from model_zoo import ModelFactory

sys.path.append("C:/Users/hliu/Desktop/DL/toolbox")
import tool


class config(object):
    """ classification model configuration """

    CLASS_NAMES = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
                   "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis",
                   "Pleural_Thickening","Hernia"]

    MODEL_NAME = "DenseNet121"
    EPOCH = 100
    BATCH_SIZE = 32
    NET_INPUT_DIM = 256
    LEARNING_RATE = 1e-3
    
    LOSS = "binary_crossentropy"
    METRICS = ['acc']
    GENERATOR_WORKERS = 8
    
    TRAIN_STEPS = 2364
    VAL_STEPS = 339

    # Either (a) Directory of images (b) CSV filepath
    #########################################################
    TRAIN = "C:/Users/hliu/Desktop/SDFN/my_train.csv"
    VAL = "C:/Users/hliu/Desktop/SDFN/my_dev.csv"
    #########################################################


def augmentation():
    """ Real-time image augmentation """
    # return iaa.SomeOf((0, 5),
    #             [    
    #             iaa.Fliplr(0.5),
    #             iaa.Flipud(0.5),
    #             iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, 
    #                     translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)}, 
    #                     rotate=(-10, 10), 
    #                     shear=(-10, 10)),
    #             iaa.CropAndPad(percent=(-0.2,0.2)),
    #             iaa.Add((-10, 10), per_channel=0.5),
    #             iaa.ContrastNormalization((0.8,1.4),per_channel=0.5),
    #             iaa.Sharpen(alpha=(0, 1.0), lightness=(0.85, 1.25)),
    #             iaa.Multiply((0.9, 1.1), per_channel=0.5),
    #             ])
    return iaa.Fliplr(0.5)


class MultipleClassAUROC(Callback):
    """Monitor mean AUROC and update model"""
    def __init__(self, auc_log_fp, sequence, class_names, workers=1):
        super(Callback, self).__init__()
        self.auc_log_fp = auc_log_fp
        self.sequence = sequence
        self.workers = workers
        self.class_names = class_names
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []

    def on_epoch_end(self, epoch, logs={}):
        print(f"\n*************** Epoch {epoch+1} ***************")
        learning_rate = float(kb.eval(self.model.optimizer.lr))
        print(f"Learning rate: {learning_rate:.7f}")
        current_auroc = []
        y_hat = self.model.predict_generator(self.sequence, workers=self.workers)
        y = self.sequence.get_y_true()  
        for i in range(len(self.class_names)):
            try:    
                score = roc_auc_score(y[:, i],y_hat[:,i])
            except ValueError:
                score = 0
            self.aurocs[self.class_names[i]].append(score)
            current_auroc.append(score)
            print("%d. %s: %f" %((i+1), self.class_names[i], score))
        mean_auroc = np.mean(current_auroc)
        if len(self.class_names)!= 1:
            print("mean auroc: %s" %(mean_auroc))
        with open(self.auc_log_fp, "a") as f:
                f.write("(epoch#%d) auroc: %f, lr: %f\n" % (
                    (epoch + 1), (mean_auroc), (learning_rate)))



class MyClassifier(object):
    def __init__(self):
        self.model = None

    def train(self, log_dir, show_model=True):
        """ Training classification model
        """
        ###################################################################################
        augs = augmentation() # how many augmentations will be used
        optimizer = SGD(lr=config.LEARNING_RATE, momentum=0.9, decay=1e-5)
        early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=9, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3,verbose=1)
        ###################################################################################

        TRAIN_CSV_FP, tmp_train = tool.prepare_dataset(config.TRAIN)
        VAL_CSV_FP,   tmp_val   = tool.prepare_dataset(config.VAL)

        SAVE_WEIGHT_FP = os.path.join(log_dir, "{epoch:03d}-{val_loss:.4f}.h5")
        AUC_LOG_FP = os.path.join(log_dir, "auc.txt")

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
            augmenter=augs,
            )

        # Validation dataset
        validation_sequence = AugmentedImageSequence(
            csv_fp=VAL_CSV_FP,
            class_names=config.CLASS_NAMES,
            batch_size=config.BATCH_SIZE,
            target_size=(config.NET_INPUT_DIM, config.NET_INPUT_DIM),
            steps=config.VAL_STEPS,
            shuffle_on_epoch_end=False,
            )

        # Load classification model
        model = ModelFactory().get_classification_model(class_num=len(config.CLASS_NAMES),
                                                        model_name=config.MODEL_NAME,
                                                        base_weights="imagenet",
                                                        input_shape=(config.NET_INPUT_DIM,config.NET_INPUT_DIM,3))
        if show_model: print(model.summary())

        model.compile(optimizer=optimizer, loss=config.LOSS, metrics=config.METRICS)

        # Callbacks
        checkpoint = ModelCheckpoint(SAVE_WEIGHT_FP, save_weights_only=False, save_best_only=False, verbose=0)
        tensorboard = TensorBoard(log_dir=os.path.join(log_dir, "logs"))
        csv_logger = CSVLogger(os.path.join(log_dir, "my_logger.csv"))
        auroc = MultipleClassAUROC(AUC_LOG_FP, validation_sequence, config.CLASS_NAMES, config.GENERATOR_WORKERS)
        callbacks = [checkpoint, tensorboard, csv_logger, early_stop, reduce_lr, auroc]

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
        Predicted probability.
        '''    
        assert self.model is not None, "Please load model"
        image = tool.read_image(image_fp,3)
        image = tool.resize_image(image,(config.NET_INPUT_DIM,config.NET_INPUT_DIM))
        image = tool.normalize(image)
        batch = np.asarray([image])
        prob = np.squeeze(self.model.predict(batch),0).tolist()
        return prob


    def write_rsult(self, test_src, save_fp, threshold=0.5):
        """ Write classification results as a file
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
                prob = self.predict(image_fp)
                target = [round(x, 5) for x in prob]
                target_str = ",".join(str(x) for x in target)
                f.write(image_fp+","+target_str+"\n")


    def compute_acc(self, csv_test, threshold):
        assert self.model is not None, "Please load model"
        df = pd.read_csv(csv_test)
        image_num = df.shape[0]
        count=0
        for idx, row in df.iterrows():
            image_fp = row["image_fp"]
            prob = self.predict(image_fp)[1]
            target = int(prob > threshold)
            if target==int(row["gt"]):
                count+=1
        acc = round(count/image_num, 4)
        print(f"Classification acc: {acc}")
