from my_classifier import MyClassifier

import sys
sys.path.append("C:/Users/hliu/Desktop/DL/toolbox")
import tool



if __name__ == "__main__":

    log_dir = "C:/Users/hliu/Desktop/DL/models/classification/run1_InceptionV3"
    my_classifier = MyClassifier()
    my_classifier.train(log_dir)	
