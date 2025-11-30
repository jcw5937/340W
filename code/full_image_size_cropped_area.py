####################################################
# Calculate the cropped area per 598 by 598 pixels #
# Calculate the full image size                    #
# Add pathology information to the results         #
####################################################

# Load the modules
import os
import re
import pandas as pd
from PIL import Image
import numpy as np

BASE_DIR = "/data/ds340w"
WORK_DIR = os.path.join(BASE_DIR, "work")

####################################################
# Calculate the cropped area per 598 by 598 pixels #
####################################################

# -------- TESTING IMAGES -------- #
def process_image_and_extract_groups(image_path):
    # Pattern for test images — now keeps ROI suffix (like _1, _1__1-1)
    pattern = r'.*(Mass|Calc)-Test_P_(\d+)_(LEFT|RIGHT)_(CC|MLO)(_[^\.]+)?\.png'
    match = re.search(pattern, image_path)

    if match:
        mass_or_calc = match.group(1)
        number       = match.group(2)
        left         = match.group(3)
        cc           = match.group(4)
        suffix       = match.group(5) if match.group(5) else ""

        result = f"{mass_or_calc}-Test_P_{number}_{left}_{cc}{suffix}"

        image = Image.open(image_path)
        image_array = np.array(image)

        nonzero_pixels = np.count_nonzero(image_array)
        total_pixels = image_array.size
        nonzero_area_percentage = (nonzero_pixels / total_pixels) * 100

        return result, image, nonzero_area_percentage

    else:
        print(f"No match found for file: {image_path}")
        return None, None, None


df_test = pd.DataFrame(columns=['name', 'file_path', 'area_percentage'])

folder_path = os.path.join(WORK_DIR, "roi_train_needed")

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        results, image, nonzero_area_percentage = process_image_and_extract_groups(image_path)
        if results:
            df_test = pd.concat(
                [df_test,
                 pd.DataFrame(
                     {'name': [results],
                      'file_path': [image_path],
                      'area_percentage': [nonzero_area_percentage]}
                 )],
                ignore_index=True
            )

print("Done for testing images")

# -------- TRAINING IMAGES -------- #
def process_image_and_extract_groups(image_path):
    # Same fix for training images
    pattern = r'.*(Mass|Calc)-Training_P_(\d+)_(LEFT|RIGHT)_(CC|MLO)(_[^\.]+)?\.png'
    match = re.search(pattern, image_path)

    if match:
        mass_or_calc = match.group(1)
        number       = match.group(2)
        left         = match.group(3)
        cc           = match.group(4)
        suffix       = match.group(5) if match.group(5) else ""

        result = f"{mass_or_calc}-Training_P_{number}_{left}_{cc}{suffix}"

        image = Image.open(image_path)
        image_array = np.array(image)

        nonzero_pixels = np.count_nonzero(image_array)
        total_pixels = image_array.size
        nonzero_area_percentage = (nonzero_pixels / total_pixels) * 100

        return result, image, nonzero_area_percentage

    else:
        print(f"No match found for file: {image_path}")
        return None, None, None


df = pd.DataFrame(columns=['name', 'file_path', 'area_percentage'])

folder_path = os.path.join(WORK_DIR, "roi_train_needed")

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        results, image, nonzero_area_percentage = process_image_and_extract_groups(image_path)
        if results:
            df = pd.concat(
                [df,
                 pd.DataFrame(
                     {'name': [results],
                      'file_path': [image_path],
                      'area_percentage': [nonzero_area_percentage]}
                 )],
                ignore_index=True
            )

print("Done for training images")

#####################################################
# Combine train + test and attach pathology         #
#####################################################

merged_df = pd.concat([df, df_test])
sorted_merged_df = merged_df.sort_values(by='name').reset_index(drop=True)

pathology = pd.read_csv(os.path.join(WORK_DIR, "all_mass_pathology.csv"))
sorted_pathology = pathology.sort_values(by='full_name').reset_index(drop=True)

sorted_merged_df['pathology'] = None

def copy_pathology(row):
    for name in sorted_pathology['full_name']:
        if name in row['name']:
            return sorted_pathology.loc[sorted_pathology['full_name'] == name,
                                        'pathology'].values[0]
    return None

sorted_merged_df['pathology'] = sorted_merged_df.apply(copy_pathology, axis=1)

sorted_merged_df.to_csv(os.path.join(WORK_DIR, "598_percentage_all.csv"))

#################################
#     Full Image Size           #
#################################

folder_path = os.path.join(WORK_DIR, "full_train_needed")

image_path_list = []
width_list = []
height_list = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.png'):
        image_path = os.path.join(folder_path, file_name)
        image_name = file_name.split('full_')[1].split('.png')[0]

        image = Image.open(image_path)
        width, height = image.size

        image_path_list.append(image_name)
        width_list.append(width)
        height_list.append(height)

df_full = pd.DataFrame({'Subject_ID': image_path_list,
                        'Width': width_list,
                        'Height': height_list})

df_full['pathology'] = df_full['Subject_ID'].apply(copy_pathology)
df_sort = df_full.sort_values(by='Subject_ID')

df_sort.to_csv(os.path.join(WORK_DIR, "heaght_width_FULL.csv"))
