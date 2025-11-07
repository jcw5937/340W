###################################
# Save the pathology information
###################################

# Load module
import pandas as pd
import re
import os

# Added for Cluster
BASE_DIR = "/data/ds340w"
WORK_DIR = os.path.join(BASE_DIR, "work")

# Read in the description data - Changed for Cluster
DATA_DIR  = os.path.join(BASE_DIR, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "mass_case_description_train_set.csv")
TEST_CSV  = os.path.join(DATA_DIR, "mass_case_description_test_set.csv")

train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

# Extract columns we need
train_need = train[['patient_id', 'pathology', 'image file path', 'cropped image file path', 'ROI mask file path']]
test_need = test[['patient_id', 'pathology', 'image file path', 'cropped image file path','ROI mask file path']]

# Merge and reindex the dataframe
merged_df = pd.concat([train_need, test_need], axis=0)
merged_df.reset_index(drop=True, inplace=True)

# Extract the names
merged_df['crop_name'] = merged_df['cropped image file path'].apply(lambda x: x.split('/')[0])
merged_df['ROI_name'] = merged_df['ROI mask file path'].apply(lambda x: x.split('/')[0])
merged_df['full_name'] = merged_df['image file path'].apply(lambda x: x.split('/')[0])

# Save the output - Changed for Cluster
merged_df.to_csv(os.path.join(WORK_DIR, 'all_mass_pathology.csv'), index=False)

