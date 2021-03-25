#!/usr/bin/env python3

# PROGRAMMER: Mike Hayes
# DATE CREATED:   3/24/2021

# Import required libraries
#Import required libraries
import pandas
import numpy as np
import json
from PIL import Image

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

import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir', type = str, default = 'flowers/train/', help = 'flowers/train/, flowers/valid/, flowers/test/')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'vgg16, densenet121')
    parser.add_argument('--save_dir', type = str, help = 'Identify where to save checkpoints')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Set the learning rate')
    parser.add_argument('--hidden_units', type = int, default = 4096, help = '4096, 500')
    parser.add_argument('--epochs', type = int, default = 5, help = 'Set the number of cycles to run')
    parser.add_argument('--device', type = str, default = "cuda", help = 'Choose cpu or gpu for training model')
    
    args = parser.parse_args()
    args.dir = str(input("dir: "))
    args.arch = str(input("architecture: "))
    args.save_dir = str(input("save director: "))
    args.learning_rate = float(input("learning rate: "))
    args.hidden_units = int(input("hidden units: "))
    args.epochs = int(input("epochs: "))
    args.device = str(input("cuda or cpu: "))
    
    return args

def get_predict_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image', type = str, default = "flowers/test/6/image_07173.jpg", help = 'Pick image to predict')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoints/checkpoint.pth', help = 'Point to desired checkpoint')
    parser.add_argument('--top_k', type = int, default = 5, help = 'Identify where to save checkpoints')
    parser.add_argument('--category_names', type = str, default = "cat_to_name.json", help = 'Point to json file')
    parser.add_argument('--device', type = str, default = "cpu", help = 'Choose cpu or gpu for training model')
    
    args = parser.parse_args()
    #args.image = str(input("Image: "))
    #args.checkpoint = str(input("Checkpoint: "))
    #args.top_k = int(input("Top K: "))
    #args.category_names = str(input("JSON File: "))
    #args.device = str(input("cuda or cpu: "))
    
    return args

def check_image_arg(image_arg):
    if image_arg is None:
        print("get_input_args had not yet been defined. No check will be performed.")
    else:
        print("Command Line Arguments:\n",
              "\n          image = ", image_arg.image, "\n     Checkpoint = ", image_arg.checkpoint,
              "\n          top_k = ", image_arg.top_k, "\n category_names = ", image_arg.category_names,
              "\n         device = ", image_arg.device)

def check_args(in_arg):
    if in_arg is None:
        print("get_input_args had not yet been defined. No check will be performed.")
    else:
        print("Command Line Arguments:\n           dir = ", in_arg.dir,
              "\n          arch = ", in_arg.arch, "\n      save dir = ", in_arg.save_dir, "\n learning rate = ", in_arg.learning_rate,
              "\n  hidden units = ", in_arg.hidden_units, "\n        epochs = ", in_arg.epochs, "\n        device = ", in_arg.device)

def data_transformer(data_dir, data_sets):
    # Define transforms for the training (including augmentation), validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    }
    return data_transforms

def load_model_classifier(arch = "vgg16", hidden_units = 4096):
    model = getattr(models,arch)(pretrained=True)
    model.name = arch
    print("Your selected network architecture is {}.".format(model.name))
    
    # Freeze model parameters to avoid backpropogation
    for param in model.parameters():
        param.requires_grad = False
    
    # Find # input features of model
    input_ft = model.classifier[0].in_features
    
    # Define classifier, loss function, and optimizer
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(input_ft, hidden_units)),
                                ('relu1', nn.ReLU()),
                                ('drop1', nn.Dropout(0.5)),
                                ('fc2', nn.Linear(hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim=1))]))
    return model, classifier

def train_image_classifier(model, dataloaders, criterion, optimizer, adj_lr, n_epochs, device, dataset_sizes):
    #capture start time to calculate duration
    start_time = time()
    
    # Initialize best models & accuracy
    best_model_weights = copy.deepcopy(model.state_dict)
    best_accuracy = 0.0
    
    # Train neural network
    print_every = 30
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
        print('-' * 10)
    
        # Cycle through train and valid datasets for each epoch
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()

            running_loss = 0.0
            running_accuracy = 0

            # Iterate through image data
            for images, labels in dataloaders[phase]:
                # Move input and label tensors to the default device
                images, labels = images.to(device), labels.to(device)

                # Zero parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    logps = model.forward(images)
                    _, preds = torch.max(logps, 1)
                    loss = criterion(logps, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Calculate statistics
                running_loss += loss.item() * images.size(0)
                running_accuracy += torch.sum(preds == labels.data)
            if phase == 'train':
                adj_lr.step() # adjust learning rate    

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_accuracy.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f}.. Acc: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))
            
            if phase == 'valid' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
                
        print()
    time_elapsed = time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy: {:4f}'.format(best_accuracy))
    
    # load best model weights
    model.load_state_dict(best_model_weights)
    
    return model

def validation(model, dataloaders, device):
# Validate training model with test data
    count = 0
    accurate = 0
    model.eval()
    with torch.no_grad():
        # Validate
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            count += labels.size(0)
            accurate += (predicted == labels).sum().item()

    print('The accuracy of the neural network model on the test images was: %d %%' % (100 * accurate / count))
    
# Function that loads a checkpoint and rebuilds the model
def load_image_classifier_model(path):
    checkpoint = torch.load(path)
    model = getattr(models,checkpoint['architecture'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    epochs = checkpoint['epoch']
    
    return model, epochs

def process_image(image):
    # Process a PIL image for use in a PyTorch model
    print(type(image))
    test_image = Image.open(image)
    original_width, original_height = test_image.size
    
    #Resize image according to shortest length
    if original_width < original_height:
        test_image.thumbnail((256,256**100))
    else:
        test_image.thumbnail((256**100, 256))
    
    #Crop image
    center = original_width/4, original_height/4
    left, top, right, bottom = center[0]-(224/2), center[1]-(224/2), center[0]+(224/2), center[1]+(224/2)
    test_image = test_image.crop((left, top, right, bottom))
    
    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_image = np.array(test_image) / 255 

    # Normalize image
    normalise_means = np.array([0.485, 0.456, 0.406])
    normalise_std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - normalise_means) / normalise_std
        
    # Set the color to the first dimension
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def predict(image_path, model, topk, device, cat_to_name):
    # Predict the class from an image file
    model.to(device)
    model.eval()
    #Convert image from numpy to torch
    image = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to("cpu")
    
    # Identify probabilities by forward passing
    logps = model.forward(image)
    
    # Linear scale probability
    linear_logps = torch.exp(logps)
    
    # Find topk results
    top_probs, top_labels = linear_logps.topk(topk)
    
    # Detach details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to class
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

