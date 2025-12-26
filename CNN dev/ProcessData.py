"""
This script is to process the data into images and labels.
Created by Jonah Boutin on 02/07/2025.
"""
import os
from scipy import io
import numpy as np


def process_data():    
    folder_path = r"/home/deeplearningtower/Documents/JonahCode/MLTestData/NewRFData"
    num_files = 1936
    image_shape = (2500,256,2)
    images = np.empty((num_files,*image_shape),dtype=np.float32)
    labels = np.empty((num_files,),dtype=np.float32)

    i = 0
    for file in os.listdir(folder_path):
        if os.fsdecode(file).endswith('.mat'):
            try:
                mat_data = io.loadmat(folder_path + '//' + file)
            except FileNotFoundError as e:
                print(e)
            else:
                frame_one = mat_data['result'][0,0]['Frame1']
                frame_two = mat_data['result'][0,0]['Frame2']
                tumor_mask = mat_data['result'][0,0]['image_information'][0,0]['tumor_mask'].astype(bool)
                youngs_modulus_matrix = mat_data['result'][0,0]['image_information'][0,0]['YM_image']

                frame_one = mat_data['output'][0,0]['field_ii_info'][0,0]['frame_one']
                frame_two = mat_data['output'][0,0]['field_ii_info'][0,0]['frame_two']
                elastography_image = mat_data['output'][0,0]['images'][0,0]['elastography_image']

                tumor_mean_YM = np.mean(youngs_modulus_matrix[tumor_mask])    
                background_mean_YM = np.mean(youngs_modulus_matrix[~tumor_mask])

                frame_one = (frame_one - np.min(frame_one)) / (np.max(frame_one) - np.min(frame_one))
                frame_two = (frame_two - np.min(frame_two)) / (np.max(frame_two) - np.min(frame_two))

                images[i] = np.stack((frame_one, frame_two), axis=-1)
                labels[i] = np.round(tumor_mean_YM/background_mean_YM,1)
                       
                i = i + 1
                print('Done File ',i)
    np.save('Data/images_final', images)
    np.save('Data/labels_final.npy', labels)
    print("Done Processing")

process_data()
