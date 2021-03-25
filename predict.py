#!/usr/bin/env python3

# PROGRAMMER: Mike Hayes
# DATE CREATED:   3/24/2021   

#Import required libraries
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

import PIL
from PIL import Image
from collections import OrderedDict
import time
import os
import copy

from req_functions import *


#------------- MAIN FILE -------------#
def main():
    start_time = time()
    
    # Define command line arguments for prediction
    prediction_arg = get_predict_args()
    check_image_arg(prediction_arg)
    image = prediction_arg.image
    print(type(image))
    
    # Load previously trained model
    model, epochs = load_image_classifier_model(prediction_arg.checkpoint)
    print(type(image))
    
    # Process selected image
    image_object = process_image(image)
    
    # Use selected cpu or gpu (cpu is sufficient for this)
    device = prediction_arg.device
    
    # Predict top k for image that was processed

    top_probs, top_labels, top_flowers = predict(image_object, model, prediction_arg.top_k, device, prediction_arg.category_names)
    
    for flower_class, prob in zip(top_flowers, top_probs):
        print("{} : {:.2%}".format(flower_class, prob))
    
    
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

# Call to main function to run the program
if __name__ == "__main__":
    main()