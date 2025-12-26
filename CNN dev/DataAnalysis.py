"""
This script is to analyze the data, and determine loss functions
Created by Jonah Boutin on 02/05/25
"""
import numpy as np
import matplotlib.pyplot as plt
images = np.load('images.npy')
labels = np.load('labels.npy')
plt.hist(labels,bins=30)
plt.title("Labels")
plt.savefig("Labels.png", dpi=300, bbox_inches="tight")