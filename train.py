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

from collections import OrderedDict
import time
from time import time
import os
import copy

from req_functions import *

#------------- MAIN FILE -------------#
def main():
    start_time = time()
    
    in_arg = get_input_args()
    check_args(in_arg)
    
    data_dir = 'flowers'
    data_sets = ['train', 'valid', 'test']
    
    # Check if gpu is available if chosen
    if in_arg.device == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    
    # Transform data sets and prepare for further processing
    data_transforms = data_transformer(data_dir, data_sets)
    
    # Load Train, Validation, and Test datasets and dataloaders
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in data_sets}
    dataset_sizes = {x: len(image_datasets[x]) for x in data_sets}
    class_names = image_datasets['train'].classes
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                   for x in data_sets}
    
    # Build Model and Classifier
    model, model.classifier = load_model_classifier(in_arg.arch, in_arg.hidden_units)
    
    # Create criterion, optimizer, adjustable learning rate
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    adj_lr = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Reduce learning rate by 0.1 every 7 epochs
    model.to(in_arg.device);
    
    # Train model
    model_ft = train_image_classifier(model, dataloaders, criterion, optimizer, adj_lr, in_arg.epochs, device, dataset_sizes)
    
    # Validate model
    validation(model, dataloaders, device)
    
    # Save mapping of classes to indices from the training dataset
    model.class_to_idx = image_datasets['train'].class_to_idx

    # Save the checkpoint
    path = "{}/checkpoint.pth".format(in_arg.save_dir)
    torch.save({
        'epoch': in_arg.epochs,
        'architecture': model.name,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    
        end_time = time()
    
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

# Call to main function to run the program
if __name__ == "__main__":
    main()