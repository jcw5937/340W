####################################
# Model development and evaluation #
####################################

################################
# Image load and visualization #
################################

# Install packages/hubs/ others needed
# pip install pydicom # taken out for cluter

# Load modules
import os
# import pydicom
from PIL import Image
import shutil
# import cv2
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import PIL
# import matplotlib.image as mpimg
import random
# from keras.utils import to_categorical
# from concurrent.futures import ThreadPoolExecutor
# from glob import glob

#### Added for clster
BASE_DIR = "/data/ds340w"
WORK_DIR = os.path.join(BASE_DIR, "work")
MODEL_DIR = os.path.join(WORK_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Read in the pathology info
dicom_data = pd.read_csv(os.path.join(WORK_DIR, 'all_mass_pathology.csv')) # changed for cluster

# Read in the image path info
# Specify the root path where the .png files are located
jpg_root_path = os.path.join(WORK_DIR, 'train_598_augmented') 
# jpg_root_path = os.path.join(WORK_DIR, 'train_598_aug_v2')

######################################################################################################
# removing augmentation because of the errors  #######################################################
######################################################################################################


# Function to get all .png file paths in a directory
def get_jpg_file_paths(directory):
    jpg_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                jpg_paths.append(os.path.join(root, file))
    return jpg_paths

# Get all .png file paths under the specified directory
jpg_paths = get_jpg_file_paths(jpg_root_path)

# Create a DataFrame with the file paths
df = pd.DataFrame({'File_Paths': jpg_paths})

# Extracting the information from the file paths and matching with dicom_data
df['ID1'] = df['File_Paths'].apply(lambda x: x.split('/')[-1] if len(x.split('/')) > 1 else '')
dicom_data['ID1'] = dicom_data['image file path'].apply(lambda x: x.split('/')[-4] if len(x.split('/')) > 1 else '')

# Assuming unique_df and jpg_paths_df are your DataFrames
for index, row in dicom_data.iterrows():
    # Check if 'image_file_path_first_part' is in 'image_file_path' of jpg_paths_df
    mask = df['ID1'].str.contains(row['ID1'])
    #mask = df['ID1'].astype(str).str.contains(str(row['ID1']), na=False) # attempted cluster fix
    #if not mask.any():
    #    continue
    # If there is a match, update 'pathology' in jpg_paths_df
    df.loc[mask, 'pathology'] = row['pathology']
    # df.loc[mask, 'full_image_name'] = row['image_file_path_first_part']

#remove the missing rows
df.dropna(subset=['pathology'], inplace=True)

# Save the df to a CSV file as all.csv
csv_path = os.path.join(WORK_DIR, 'all.csv') # changed for cluster
df.to_csv(csv_path, index=False)

###################################
# Label the pathology information #
###################################

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Map labels to binary values
label_mapping = {'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1}
df['label'] = df['pathology'].map(label_mapping)

# Get the images before augmentation
def check_filename(filename):
    basename = os.path.basename(filename)
    return basename.endswith('1-1_1.png') or basename.endswith('1-2_1.png') # adjusted to match filenames on cluster

# Apply the function to filter rows
original_df = df[df['File_Paths'].apply(check_filename)]
# Trying a new thing on the cluster
# original_df = df  # use all images (original + augmented) on the cluster


###################
# Build the model #
###################

# Install tensorflow-hub and timm
# pip install tensorflow-hub # removed for clutsre
# pip install timm # removed for clutser

# Model structure
import torch
import torch.nn as nn
import torch.nn.functional as F
# import timm
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Define the additional convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),  # Change input channels to 1
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=1),  # Change to 3 output channels to match Xception input
            nn.LeakyReLU(0.1)
        )
        
#         # Load the pretrained Xception model
#         self.feature_extractor = timm.create_model('legacy_xception', pretrained=True)
        
#         # Remove the last two layers of the Xception model
#         self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])


        # load resnet pretrained model - feel free to adjust based on your needs
        self.feature_extractor = models.resnet50(weights=ResNet50_Weights.DEFAULT)

          # Remove the last fully connected layer of ResNet-50
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

#         self.feature_extractor = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        
#         # Remove the last fully connected layer of DenseNet-121
#         self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # Add the desired layers
        self.global_pooling = nn.AdaptiveAvgPool2d(1)  # Global average pooling

        self.fc1 = nn.Linear(2048, 512) # For resnet50 and Xception

        self.relu = nn.ReLU()  # Changed LeakyReLU to ReLU
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.conv_layers(x)
        # print("After conv_layers shape:", x.shape)
        
        # Forward pass through the feature extractor
        x = self.feature_extractor(x)
        
        # Forward pass through the added layers
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)  # Moved dropout after ReLU in fc2
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x

# Create an instance of the model
model = CustomModel()

# Print model summary
# print(model)

##########################
# Image data processing  #
##########################

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # Assuming the image paths are in the first column
        image = Image.open(img_name)  # Open image without converting to 'L'
        image = np.array(image)  # Convert image to numpy array
        image = image.astype(np.float32)  # Convert to float32
        
        label = self.data.iloc[idx, 3]  # Assuming the labels are in the fourth column
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
  
# Define the transformation pipeline - adjust based on which data you plan to use
transform_ = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    #transforms.Lambda(lambda x: x / 65535.0),  # Normalize tensor by dividing by 65535.0 - trying taking this out
    # transforms.CenterCrop(299),
    # transforms.Resize((299, 299))  # Ensure the final size is 448x448
    # transforms.CenterCrop(448),  # Center crop the image to 448x448
    # transforms.Resize((448, 448))  # Ensure the final size is 448x448
    transforms.CenterCrop(224),
    transforms.Resize((224, 224))  # Ensure the final size is 448x448

])
from sklearn.model_selection import train_test_split

# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

# Split the dataset into train and test sets, get all the images for training
#train_data, temp_data = train_test_split(original_df, test_size=0.2)
train_data, temp_data = train_test_split(original_df, test_size=0.2, random_state=42)
# Function to replicate rows and modify File_Paths
def replicate_rows(df, n_replicates):
    # Create an empty list to collect new rows
    new_rows = []
    
    # Iterate over each row in the original DataFrame
    for _, row in df.iterrows():
        # Add the original row once
        new_rows.append(row)
        
        # Add replicated rows with modified File_Paths
        for i in range(1, n_replicates + 1):
            new_row = row.copy()
            new_row['File_Paths'] = new_row['File_Paths'].replace('.png', f'_{i}.png')
            new_rows.append(new_row)
    
    # Convert list of rows to DataFrame
    return pd.DataFrame(new_rows)

# Replicate each row 5 times with modified File_Paths
# train_data_augmented = replicate_rows(train_data, 5) ********* took this out for cluster

# Split the train set into train and validation sets
# val_data, test_data = train_test_split(temp_data, test_size=1/2)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

# Create datasets and dataloaders for train, validation, and test sets
#train_dataset_ = CustomDataset(train_data_augmented, transform=transform_) #this line has been changed for the cluster:
train_dataset_ = CustomDataset(train_data, transform=transform_)
train_loader_ = DataLoader(train_dataset_, batch_size=32, shuffle=True)

val_dataset = CustomDataset(val_data, transform=transform_)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # No need to shuffle validation data

test_dataset = CustomDataset(test_data, transform=transform_)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No need to shuffle test data

#################
# Run the model #
#################

import torch.optim as optim

# Define loss function and optimizer
criterion = nn.BCELoss()  # Use binary cross-entropy loss function since the labels are binary
# optimizer = optim.Adam(model.parameters(), lr=0.000003)  # Adjust the learning rate based on your needed
optimizer = optim.Adam(model.parameters(), lr=0.000004) # new learning rate attempts

# Move the model and data to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20 # It could be increases based on your model's performance

for epoch in range(num_epochs):
    model.train()  # Set the model to train mode
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Inside the training loop
    for images, labels in train_loader_:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs.squeeze(), labels.float())  # Squeeze to remove extra dimension

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        #predicted_labels = (outputs > 0.5).float()  # Threshold at 0.5 for binary classification
        predicted_labels = (outputs > 0.35).float() # trying a new threshold
        predicted_labels_int = predicted_labels.view(-1).long()
        total_correct += (predicted_labels_int == labels).sum().item()
        total_samples += labels.size(0)
          
    # Calculate accuracy and loss for training set
    train_accuracy = total_correct / total_samples
    train_loss = running_loss / len(train_loader_)
    
    # Validation
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0.0
    val_total_correct = 0
    val_total_samples = 0
    
    with torch.no_grad():  # No need to calculate gradients during validation
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)  # Move data to GPU

            # Forward pass
            val_outputs = model(val_images)

            # Calculate loss
            val_loss = criterion(val_outputs.squeeze(), val_labels.float())  # Squeeze to remove extra dimension
            val_running_loss += val_loss.item()

            # Calculate accuracy
            val_predicted_labels = (val_outputs > 0.45).float()  # Threshold at 0.5 for binary classification
            val_predicted_labels_int = val_predicted_labels.view(-1).long()
            val_total_correct += (val_predicted_labels_int == val_labels).sum().item()
            val_total_samples += val_labels.size(0)

    # Calculate accuracy and loss for validation set
    val_accuracy = val_total_correct / val_total_samples
    val_loss = val_running_loss / len(val_loader)

    # Print the training and validation results for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    # Save model checkpoints for the last 10 epochs
    if epoch >= num_epochs - 10: # The number could be adjusted based on your needs
        checkpoint_name = os.path.join(MODEL_DIR, f'model_epoch_{epoch + 1}.pth') #changed for cluster
        torch.save(model.state_dict(), checkpoint_name)
        print(f"Checkpoint saved: {checkpoint_name}")

####################
# Model evaluation #
####################

######################
#1. Calculate the accuracy, precision, recall, AUROC, and F1 score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

# # Assuming you have defined your device as 'cuda' if available, else 'cpu'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the best model state dictionary based on the validation accuracy
model.load_state_dict( # changed for cluster
    torch.load(os.path.join(MODEL_DIR, f'model_epoch_{num_epochs}.pth'), map_location=device)
) # You will want to update this based on your training checkpoints' performances

# Set the model to evaluation mode
model.eval()

# Lists to store true labels and predicted probabilities
true_labels = []
predicted_probs = []

# Loop through the test dataset
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)

        # Forward pass
        outputs = model(test_images)

        # Calculate predicted probabilities
        predicted_probs.extend(outputs.cpu().numpy())

        # Convert labels to numpy array and append to true_labels list
        true_labels.extend(test_labels.cpu().numpy())

# Convert true_labels and predicted_probs to numpy arrays
true_labels = np.array(true_labels)
predicted_probs = np.array(predicted_probs)

# Calculate predicted labels
predicted_labels = (predicted_probs > 0.45).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
roc_auc = roc_auc_score(true_labels, predicted_probs)

#different thresholds testing
print("\n=== Threshold sweep (no bootstrap, quick view) ===")
for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    thr_pred = (predicted_probs >= thr).astype(int)
    acc_thr = accuracy_score(true_labels, thr_pred)
    prec_thr = precision_score(true_labels, thr_pred, zero_division=0)
    rec_thr = recall_score(true_labels, thr_pred, zero_division=0)
    f1_thr = f1_score(true_labels, thr_pred, zero_division=0)
    pos_rate = thr_pred.mean()
    print(
        f"thr={thr:.2f} | acc={acc_thr:.3f} "
        f"prec={prec_thr:.3f} rec={rec_thr:.3f} f1={f1_thr:.3f} "
        f"pos_rate={pos_rate:.3f}"
    )


# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

######################
#2. calculate 95 CI%

# Define the number of iterations for bootstrap
n_iterations = 1000  # Can be adjusted as needed

# Define lists to store evaluation metric values after bootstrap sampling
accuracy_bootstrap = []
precision_bootstrap = []
recall_bootstrap = []
f1_bootstrap = []
roc_auc_bootstrap = []

# Perform bootstrap resampling and calculate evaluation metric values
for _ in range(n_iterations):
    # Generate bootstrap samples
    bootstrap_indices = np.random.choice(len(true_labels), size=len(true_labels), replace=True)
    bootstrap_true_labels = true_labels[bootstrap_indices]
    bootstrap_predicted_labels = predicted_labels[bootstrap_indices]
    bootstrap_predicted_probs = predicted_probs[bootstrap_indices]

    # Calculate evaluation metric values
    bootstrap_accuracy = accuracy_score(bootstrap_true_labels, bootstrap_predicted_labels)
    bootstrap_precision = precision_score(bootstrap_true_labels, bootstrap_predicted_labels)
    bootstrap_recall = recall_score(bootstrap_true_labels, bootstrap_predicted_labels)
    bootstrap_f1 = f1_score(bootstrap_true_labels, bootstrap_predicted_labels)
    bootstrap_roc_auc = roc_auc_score(bootstrap_true_labels, bootstrap_predicted_probs)

    # Append evaluation metric values to the corresponding lists
    accuracy_bootstrap.append(bootstrap_accuracy)
    precision_bootstrap.append(bootstrap_precision)
    recall_bootstrap.append(bootstrap_recall)
    f1_bootstrap.append(bootstrap_f1)
    roc_auc_bootstrap.append(bootstrap_roc_auc)

# Calculate the upper and lower bounds of the 95% CI
accuracy_ci_bootstrap = np.percentile(accuracy_bootstrap, [2.5, 97.5])
precision_ci_bootstrap = np.percentile(precision_bootstrap, [2.5, 97.5])
recall_ci_bootstrap = np.percentile(recall_bootstrap, [2.5, 97.5])
f1_ci_bootstrap = np.percentile(f1_bootstrap, [2.5, 97.5])
roc_auc_ci_bootstrap = np.percentile(roc_auc_bootstrap, [2.5, 97.5])

# Print the 95% CI results calculated using the bootstrap method
print(f"95% CI for Accuracy (Bootstrap): {accuracy_ci_bootstrap}")
print(f"95% CI for Precision (Bootstrap): {precision_ci_bootstrap}")
print(f"95% CI for Recall (Bootstrap): {recall_ci_bootstrap}")
print(f"95% CI for F1 Score (Bootstrap): {f1_ci_bootstrap}")
print(f"95% CI for ROC AUC (Bootstrap): {roc_auc_ci_bootstrap}")

######################
#3. plot confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

conf_matrix # Print the confusion matrix in number as well.

# Calculate row-wise sums
row_sums = conf_matrix.sum(axis=1, keepdims=True)

# Calculate percentages based on row sums
conf_matrix_percent_row = conf_matrix / row_sums * 100

# Create annotations with both count and percentage values
annotations = [f"{conf_matrix[i, j]:d}\n({conf_matrix_percent_row[i, j]:.2f}%)" for i in range(conf_matrix.shape[0]) for j in range(conf_matrix.shape[1])]
annotations = np.array(annotations).reshape(conf_matrix.shape[0], conf_matrix.shape[1])

# Define custom color palette with starting color #6da9ed and ending color #eb6a4d
#colors = sns.diverging_palette(100, 5, s=80, l=60, as_cmap=True)
colors = sns.diverging_palette(80, 5, s=70, l=80, as_cmap=True)


# Plot confusion matrix with annotations and custom colors
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=annotations, fmt="", cmap=colors, cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Add custom labels
plt.xticks(ticks=[0.5, 1.5], labels=["Benign", "Malignant"])
plt.yticks(ticks=[0.5, 1.5], labels=["Benign", "Malignant"])

plt.savefig('confusion_matrix.pdf', dpi=1000)
plt.show()

######################
#4. plot ROC curve

from sklearn.metrics import roc_curve, auc

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Set Seaborn style and color palette
sns.set_style("white")
sns.set_palette(["#e090b5", "#e6cd73"])

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Save the plot as PDF with 1000dpi
plt.savefig('roc_curve.pdf', dpi=1000)
plt.show()

#################################
# INNOVATION: SHAP explanations #
#################################


import shap  # make sure to pip install shap on teh cluster


# Ensure model is in eval mode and on the right device
model.eval()
model.to(device)

print("\n=== Building SHAP background set ===")


# ---- 1) Build a small background set from training data ----
# Use up to 50 training images as background
bg_n = min(50, len(train_data))
background_df = train_data.sample(n=bg_n, random_state=0).reset_index(drop=True)


background_dataset = CustomDataset(background_df, transform=transform_)
background_loader = DataLoader(background_dataset, batch_size=10, shuffle=False)


background_batches = []
for imgs, _ in background_loader:
   background_batches.append(imgs.to(device))


background = torch.cat(background_batches, dim=0)  # shape: [bg_n, 1, 224, 224]
print(f"Background tensor shape: {background.shape}")


# ---- 2) Pick some test images to explain (20 malignant, 20 benign if possible) ----
print("=== Building explanation set from test_data ===")


malignant_df = test_data[test_data['label'] == 1].sample(20)
benign_df    = test_data[test_data['label'] == 0].sample(20)
explain_df   = pd.concat([malignant_df, benign_df], ignore_index=True)


print(f"Number of images selected for explanation: {len(explain_df)}")


explain_dataset = CustomDataset(explain_df, transform=transform_)
explain_loader  = DataLoader(explain_dataset, batch_size=8, shuffle=False)


# ---- 3) Create GradientExplainer ----
print("=== Initializing SHAP GradientExplainer ===")
explainer = shap.GradientExplainer(model, background)


# ---- 4) Compute SHAP values for the explanation set ----
print("=== Computing SHAP values (this can take a few minutes) ===")


all_shap_values = []
all_images = []


for imgs, lbls in explain_loader:
   imgs = imgs.to(device)
   shap_vals_list = explainer.shap_values(imgs)
   # For a single-output sigmoid model, shap_values returns a list with one array
   shap_vals = shap_vals_list[0]  # shape: [batch, 1, 224, 224]


   # all_shap_values.append(shap_vals.cpu().numpy()) # this line fails do to a cpu error
   if isinstance(shap_vals, torch.Tensor):
       shap_vals = shap_vals.detach().cpu().numpy()
   all_shap_values.append(shap_vals)


   all_images.append(imgs.cpu().numpy())


all_shap_values = np.concatenate(all_shap_values, axis=0)  # [N, 1, 224, 224]
all_images      = np.concatenate(all_images, axis=0)       # [N, 1, 224, 224]


print(f"SHAP values shape: {all_shap_values.shape}")
print(f"Image batch shape: {all_images.shape}")


print(f"SHAP values shape: {all_shap_values.shape}")
print(f"Image batch shape: {all_images.shape}")
print("SHAP ndim:", all_shap_values.ndim)
print("Image ndim:", all_images.ndim)
print("Example SHAP[0] shape:", all_shap_values[0].shape)
print("Example IMG[0] shape:", all_images[0].shape)


# ---- 5) Save a few overlay plots to disk ----
print("=== Saving SHAP overlay images ===")


output_dir = os.path.join(WORK_DIR, "shap_outputs")
os.makedirs(output_dir, exist_ok=True)


# Only save as many as we actually have SHAP values for
num_to_save = min(10, all_images.shape[0], all_shap_values.shape[0])
print(f"Saving {num_to_save} SHAP examples")


for i in range(num_to_save):
   # all_images: (40, 1, 224, 224) -> take channel 0 -> (224, 224)
   img = all_images[i, 0, :, :]


   # all_shap_values: (5, 224, 224, 1) -> take channel 0 -> (224, 224)
   smap = all_shap_values[i, :, :, 0]


   plt.figure(figsize=(6, 3))


   # Original image
   plt.subplot(1, 2, 1)
   plt.imshow(img, cmap="gray", aspect="equal")
   plt.axis("off")
   plt.title("Original")


   # SHAP overlay
   plt.subplot(1, 2, 2)
   plt.imshow(img, cmap="gray", aspect="equal")
   plt.imshow(smap, cmap="jet", alpha=0.5, aspect="equal")
   plt.axis("off")
   plt.title("SHAP overlay")


   plt.tight_layout()
   out_path = os.path.join(output_dir, f"shap_example_{i:02d}.png")
   plt.savefig(out_path, dpi=200)
   plt.close()


   print(f"Saved {out_path}")

print("=== SHAP explanation generation finished ===")
