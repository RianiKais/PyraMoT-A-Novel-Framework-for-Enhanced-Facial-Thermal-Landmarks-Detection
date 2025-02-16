# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:31:42 2023

@author: kais-
"""

# import necessary modules
import fnv
import fnv.reduce
import fnv.file
import numpy as np
import os
import csv
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the CSV file containing the list of videos and frames
csv_file_path = 'E:/frames.csv'

# Path to the directory containing the video files
video_dir_path = 'E:/Segmented_S8/'

# Path to the directory where the extracted frames will be saved
output_dir_path = 'E:/images_code/'

if not os.path.exists(video_dir_path):
    print("File does not exist at path:", video_dir_path)
    exit(1)


def save_image(data, output_file_path):
    plt.imsave(output_file_path, data, cmap='afmhot')

# Open the CSV file and read the list of videos and frames
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        # Extract the video name and frames from the CSV row
        video_name = row[0] + '.ats'
        frames = row[1:]
        frames = [int(frame_num) for frame_num in frames if frame_num]

        # Create a subdirectory with the name of the video file to save the images
        video_output_dir_path = output_dir_path
        #video_output_dir_path = os.path.join(output_dir_path, video_name.split('.')[0])
        #if not os.path.exists(video_output_dir_path):
            #os.makedirs(video_output_dir_path)

        # Open the video file using the FLIR Science File SDK
        video_path = os.path.join(video_dir_path, video_name)
        im = fnv.file.ImagerFile(video_path)

        # Extract the desired frames from the video and save them as individual image files
        for frame_num in frames:
            # Convert the frame number from string to integer
            frame_num = int(frame_num)

            # Get the specified frame from the video using the FLIR Science File SDK
            im.get_frame(frame_num)
            frame_data = im.final
            data = np.array(im.final, copy=False).reshape((im.height, im.width))

            # Save the image file
            output_file_path = os.path.join(video_output_dir_path,video_name.split('.')[0] + '_' + f'{frame_num}.png')
            save_image(data, output_file_path)

        # Dispose of the ImagerFile object to free up memory
        im = None

# Create an ImageDataGenerator object to load the saved images for the deep learning model
image_data_generator = ImageDataGenerator(rescale=1./255)

# Load the images from the directory using the ImageDataGenerator object
image_size = (512, 640)
batch_size = 32
data_generator = image_data_generator.flow_from_directory(
    output_dir_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,
    color_mode='grayscale'
)
