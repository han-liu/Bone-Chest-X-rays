import os
import importlib
import keras.backend as K
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout, Activation, Reshape
from keras.layers.merge import concatenate
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, concatenate
from keras.layers import BatchNormalization, Conv2D, GlobalAveragePooling2D
from keras.models import Model

"""
# Models available in this ModelFactory:
# 
# "VGG16", "VGG19", 
# "DenseNet121", "ResNet50", 
# "InceptionV3", "InceptionResNetV2", 
# "NASNetMobile", "NASNetLarge"
"""

class ModelFactory:

    def __init__(self):
        self.models_ = dict(
            VGG16 = dict(input_shape=(224, 224, 3),
                         module_name="vgg16",
                         last_conv_layer="block5_conv3"),

            VGG19 = dict(input_shape=(224, 224, 3),
                         module_name="vgg19",
                         last_conv_layer="block5_conv4"),

            DenseNet121 = dict(input_shape=(224, 224, 3),
                               module_name="densenet",
                               last_conv_layer="bn"),

            ResNet50 = dict(input_shape=(224, 224, 3),
                            module_name="resnet50",
                            last_conv_layer="activation_49"),

            InceptionV3 = dict(input_shape=(299, 299, 3),
                               module_name="inception_v3",
                               last_conv_layer="mixed10"),

            InceptionResNetV2 = dict(input_shape=(299, 299, 3),
                                     module_name="inception_resnet_v2",
                                     last_conv_layer="conv_7b_ac"),

            NASNetMobile = dict(input_shape=(224, 224, 3),
                                module_name="nasnet",
                                last_conv_layer="activation_188"),

            NASNetLarge = dict(input_shape=(331, 331, 3),
                               module_name="nasnet",
                               last_conv_layer="activation_260"))

        self.model_names = ["VGG16", 
                          "VGG19",
                          "DenseNet121",
                          "ResNet50",
                          "InceptionV3",
                          "InceptionResNetV2",
                          "NASNetMobile",
                          "NASNetLarge"]

    def get_last_conv_layer(self, model_name):
        return self.models_[model_name]["last_conv_layer"]

    def get_input_size(self, model_name):
         return self.models_[model_name]["input_shape"][:2]

    def get_base_model(self, model_name, base_weights, input_shape):
        assert model_name in self.model_names
        base_model_class = getattr(importlib.import_module("keras.applications."\
            + self.models_[model_name]['module_name']), model_name)
        if input_shape is None: 
            input_shape = self.models_[model_name]["input_shape"]
        img_input = Input(shape=input_shape)
        base_model = base_model_class(include_top=False,
                                      input_tensor=img_input,
                                      input_shape=input_shape,
                                      weights=base_weights)
        return base_model

    def get_classification_model(self, class_num, model_name,
        base_weights="imagenet", input_shape=None):
        base_model = self.get_base_model(model_name, base_weights, input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x) 
        predictions = Dense(class_num, activation="sigmoid", name="predictions")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model


    def get_regression_model(self, class_num, model_name,
        base_weights="imagenet", input_shape=None):
        base_model = self.get_base_model(model_name, base_weights, input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x) 
        x = Dense(1024, activation="tanh")(x)
        x = Dropout(0.25)(x) 
        predictions = Dense(class_num, activation="linear", name="predictions")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model


    def boneage_winner(self, class_num, model_name,
        base_weights="imagenet", input_shape=None):

        base_model = self.get_base_model(model_name, base_weights, input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        gender_input = Input(shape=(1,), name="gender_input")
        y = Dense(32)(gender_input)

        concat = concatenate([x, y], name='concatenation_layer')
        z = Dense(1024, activation="relu")(concat)
        z = Dropout(0.25)(z)
        # z = Dense(500, activation="relu")(z)
        # z = Dropout(0.1)(z)
        predictions = Dense(class_num, activation="linear", name="predictions")(z)

        model = Model(inputs=[base_model.input, gender_input], outputs=predictions)
        return model



def LungSegNet(inp_shape, k_size=3):
    """ Lung segmentation model: U-Net
    """
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend)
    data = Input(shape=inp_shape)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(data)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool1)
    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool2)
    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(pool3)
    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(pool4)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up1)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv6)
    merged1 = concatenate([conv4, conv6], axis=merge_axis)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged1)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up2)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv7)
    merged2 = concatenate([conv3, conv7], axis=merge_axis)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged2)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(up3)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv8)
    merged3 = concatenate([conv2, conv8], axis=merge_axis)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(merged3)

    up4 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(up4)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv9)
    merged4 = concatenate([conv1, conv9], axis=merge_axis)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(merged4)

    conv10 = Convolution2D(filters=1, kernel_size=k_size, padding='same', activation='sigmoid')(conv9)

    output = conv10
    model = Model(data, output)
    return model



def loadUnet(weights_path, im_shape=(256,256)):
    """ Load lung segmentation Network
    """
    if not os.path.isfile(weights_path):
        raise Exception("Weights file does not exist")
        return
    if im_shape[0]!= im_shape[1]:
        raise Exception("im_shape[0] does not equal to im_shape[1]")
        return
    Unet = LungSegNet((im_shape[0],im_shape[1],1))
    Unet.load_weights(weights_path)
    return Unet



def buildSDFN(model1, model2, class_num):
    """ Segmentation-based Deep Fusion Network
    """
    inputs1 = model1.layers[0].output
    inputs2 = model2.layers[0].output
    
    for layer1 in model1.layers:
        layer1.trainable=False
    
    for layer2 in model2.layers:
        layer2.name = layer2.name + "_2"
        layer2.trainable=False

    glob_pool1 = model1.get_layer("avg_pool").output
    glob_pool2 = model2.get_layer("avg_pool_2").output
    concat = concatenate([glob_pool1, glob_pool2], name='concatenation_layer')
    fc = Dense(class_num, activation="sigmoid", name="predictions")(concat)
    predictions = Flatten()(fc)
    SDFN = Model(inputs=[inputs1, inputs2], outputs=predictions)
    return SDFN



def loadSDFN(weights_path, class_num):
    """ Load Segmentation-based Deep Fusion Network
    """
    if not os.path.isfile(weights_path):
        raise Exception("Weights file does not exist")
        return
    model1 = ModelFactory().get_model(14, "DenseNet121")
    model2 = ModelFactory().get_model(14, "DenseNet121")
    SDFN = buildSDFN(model1, model2, class_num)
    SDFN.load_weights(weights_path)
    return SDFN
