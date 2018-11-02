import cv2, os, sys
import numpy as np
from keras import backend as kb
sys.path.append("C:/ChestXray_DL/MyCheXnet")
from utility import get_PILimage, process_PILimage


def heatmap(model, activated_class, image_fp, output_fp, activated_threshold=0):
    
    def get_output_layer(model, layer_name):
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        layer = layer_dict[layer_name]
        return layer
    # Process the image as the classifier input
    PILimage = get_PILimage(image_fp)
    image = np.asarray(PILimage)
    processed_image = process_PILimage(PILimage)
    # Get the weights between average pooling and fc_layer
    class_weights = model.get_layer("predictions").get_weights()[0] 
    # Get the 7x7x1024 feature maps
    final_conv_layer = get_output_layer(model, "bn") 
    get_final_conv = kb.function([model.layers[0].input], [final_conv_layer.output])
    conv_outputs = get_final_conv([processed_image])[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
    for i, w in enumerate(class_weights[:,activated_class]):
        cam += w * conv_outputs[:, :, i]   
    
    # Remove negative activations (optional) 
    cam[cam<0] = 0
    
    # Normalize the heatmap
    if np.max(cam)!=0:
        cam/= np.max(cam)
    cam = cv2.resize(cam,(image_ori.shape[1],image_ori.shape[0]))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    output_img = image + heatmap * 0.3  

    # Draw a rectangle on the most activated region at an activated_threshold
    x, y = [], []
    for h in range(image_ori.shape[0]):
        for w in range(image_ori.shape[1]):
            if cam[h, w] > activated_threshold * 255:
                xx.append(h)
                yy.append(w)           
    if x and y:
        top, bot, left, right = min(x), max(x), min(y), max(y)
        cv2.rectangle(output_img, (left, top), (right, bot), (0, 0, 255), 2)

    cv2.imwrite(output_fp, output_img)
    print("Successfully created heatmap at ", output_fp)
