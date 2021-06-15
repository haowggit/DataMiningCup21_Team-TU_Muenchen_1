import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder(os.path.join(os.getcwd(), 'Cover_Pics_Ext_Final'), transform=transform)
