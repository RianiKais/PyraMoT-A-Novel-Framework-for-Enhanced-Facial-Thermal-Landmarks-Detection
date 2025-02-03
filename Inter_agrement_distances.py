import pandas as pd
import numpy as np
import math

def ccc(landmark1, landmark2):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((landmark1 - landmark1.mean()) * (landmark2 - landmark2.mean())) / landmark1.shape[0]
    var1 = np.var(landmark1)
    var2 = np.var(landmark2)
    mean_diff = landmark1.mean() - landmark2.mean()
    
    if var1 == 0 and var2 == 0:
        return np.nan
    else:
        rhoc = 2 * sxy / (var1 + var2 + mean_diff**2)
        return rhoc

def r(landmark1, landmark2):
    ''' Pearson Correlation Coefficient'''
    sxy = np.sum((landmark1 - landmark1.mean()) * (landmark2 - landmark2.mean())) / landmark1.shape[0]
    std1 = np.std(landmark1)
    std2 = np.std(landmark2)
    
    if std1 == 0 or std2 == 0:
        return np.nan
    else:
        rho = sxy / (std1 * std2)
        return rho

# Read the CSV files for each annotator
annotator1_df = pd.read_csv('F:/5050_annotator2_extracted.csv')
annotator2_df = pd.read_csv('F:/5050_annotator1_extracted.csv')

# Initialize a dictionary to store normalization factors for each image
normalization_factors = {}

# Iterate over unique image names and calculate the normalization factor for each image
for image in annotator1_df['image'].unique():
    image_df_annotator1 = annotator1_df[(annotator1_df['image'] == image)]
    image_df_annotator2 = annotator2_df[(annotator2_df['image'] == image)]
    
    max_x_annotator1 = image_df_annotator1['cx'].max()
    min_x_annotator1 = image_df_annotator1['cx'].min()
    max_y_annotator1 = image_df_annotator1['cy'].max()
    min_y_annotator1 = image_df_annotator1['cy'].min()
    
    max_x_annotator2 = image_df_annotator2['cx'].max()
    min_x_annotator2 = image_df_annotator2['cx'].min()
    max_y_annotator2 = image_df_annotator2['cy'].max()
    min_y_annotator2 = image_df_annotator2['cy'].min()
    
    max_distance_annotator1 = math.sqrt((max_x_annotator1 - min_x_annotator1)**2 + (max_y_annotator1 - min_y_annotator1)**2)
    max_distance_annotator2 = math.sqrt((max_x_annotator2 - min_x_annotator2)**2 + (max_y_annotator2 - min_y_annotator2)**2)
    
    normalization_factors[image] = (max_distance_annotator1, max_distance_annotator2)

# Normalize the coordinates based on the normalization factors for each image
annotator1_df['x_normalized'] = annotator1_df.apply(lambda row: row['cx'] / normalization_factors[row['image']][0], axis=1)
annotator1_df['y_normalized'] = annotator1_df.apply(lambda row: row['cy'] / normalization_factors[row['image']][0], axis=1)

annotator2_df['x_normalized'] = annotator2_df.apply(lambda row: row['cx'] / normalization_factors[row['image']][1], axis=1)
annotator2_df['y_normalized'] = annotator2_df.apply(lambda row: row['cy'] / normalization_factors[row['image']][1], axis=1)

# Calculate NED, CCC, Pearson coefficient, and nRMSE for each landmark
ned_list = []
ccc_list = []
pearson_list = []
nrmse_list = []

for i in range(len(annotator1_df)):
    x1_normalized = annotator1_df.loc[i, 'x_normalized']
    y1_normalized = annotator1_df.loc[i, 'y_normalized']
    x2_normalized = annotator2_df.loc[i, 'x_normalized']
    y2_normalized = annotator2_df.loc[i, 'y_normalized']

    x1 = annotator1_df.loc[i, 'cx']
    y1 = annotator1_df.loc[i, 'cy']
    x2 = annotator2_df.loc[i, 'cx']
    y2 = annotator2_df.loc[i, 'cy']    
    
    ned = math.sqrt((x1_normalized - x2_normalized)**2 + (y1_normalized - y2_normalized)**2)
    ccc_value = ccc(np.array([x1, y1]), np.array([x2, y2]))
    pearson_value = r(np.array([x1, y1]), np.array([x2, y2]))
    nrmse = math.sqrt(((x1 - x2) / normalization_factors[annotator1_df.loc[i, 'image']][0])**2 + 
                      ((y1 - y2) / normalization_factors[annotator1_df.loc[i, 'image']][0])**2)
    
    ned_list.append(ned)
    ccc_list.append(ccc_value)
    pearson_list.append(pearson_value)
    nrmse_list.append(nrmse)

# Calculate the average NED, CCC, and Pearson coefficient
average_ned = sum(ned_list) / len(ned_list)
average_ccc = np.nanmean(ccc_list)
average_pearson = np.nanmean(pearson_list)
average_nrmse = sum(nrmse_list) / len(nrmse_list)

# Print the results
print("Average NED:", average_ned)
print("Average Concordance Correlation Coefficient (CCC):", average_ccc)
print("Average Pearson correlation:", average_pearson)
print("Average nRMSE:", average_nrmse)
