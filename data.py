import os
import PIL.Image
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import utils

class CaptchaDataset(Dataset):
    """
    A PyTorch dataset class for loading CAPTCHA images and their corresponding labels.
    """
    def __init__(self, csv_path, root_dir, transform=None):
        """
        Initializes the dataset by reading image paths and labels from a CSV file.

        Args:
            csv_path (str): Path to the CSV file containing image paths and labels.
            root_dir (str): Root directory containing the images.
            transform (callable, optional): Transformations to apply to the images.
        """
        full_data = pd.read_csv(csv_path)
        
        # Filter data to include only images from the specified root directory
        self.data = full_data[full_data['image_path'].str.startswith(os.path.basename(root_dir.strip('/')))]
        
        self.root_dir = root_dir
        self.transform = transform
        
        # Character mapping for encoding and decoding labels
        self.char_map = {str(i): i for i in range(10)}  # Map digits 0-9 to label indices
        self.inv_char_map = {v: k for k, v in self.char_map.items()}  # Reverse mapping

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)
    
    def preprocessing(self, image_input, min_contour_area=0):
        """
        Applies preprocessing to remove noise and extract relevant features.
        
        Steps:
        1. Converts image to grayscale (if not already).
        2. Applies Gaussian blur to reduce noise.
        3. Uses adaptive thresholding to segment the characters.
        4. Finds and filters contours based on area.
        5. Creates a mask to keep only the relevant parts.

        Args:
            image_input (str | PIL.Image): Image file path or PIL image.
            min_contour_area (int, optional): Minimum area for a contour to be kept. Default is 0.
        
        Returns:
            tuple: (Binary mask as PIL.Image, Masked image as PIL.Image)
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_input, PIL.Image.Image):
            image = np.array(image_input)

        # Apply Gaussian Blur to smooth out noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Adaptive thresholding to highlight text regions
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area to remove small noise
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

        # Create a blank mask to store only relevant regions
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)  # Fill detected contours

        # Apply mask to extract only character regions
        final_mask = np.logical_and(mask, thresh).astype(np.uint8) * 255
        masked_image = cv2.bitwise_and(image, image, mask=final_mask)

        return Image.fromarray(final_mask), Image.fromarray(masked_image)

    def __getitem__(self, idx):
        """
        Loads and returns a single sample (image, label, mask) from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: {'image': Tensor, 'mask': Tensor, 'label': Tensor, 'label_length': Tensor, 'label_str': str}
        """
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0].split('/')[-1])
        label_str = f"{self.data.iloc[idx, 1]:06d}"  # Ensure label is always 6 digits

        # Convert label string to a tensor of character indices
        label_indices = [self.char_map[c] for c in label_str]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        image = Image.open(img_name).convert("L")  # Convert to grayscale
        
        # Preprocess image to enhance digits
        mask, masked_image = self.preprocessing(image, min_contour_area=50)
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {
            'image': image,  # Original image tensor
            'mask': mask,    # Processed binary mask tensor
            'label': label_tensor,  # Encoded label tensor
            'label_length': torch.tensor(len(label_str), dtype=torch.long),  # Label length
            'label_str': label_str  # Original label as a string
        }


def get_transforms(config, val_dataset=False):
    """
    Creates and returns a set of image transformations based on the config settings.

    Args:
        config (dict): Dictionary containing transformation settings.
        val_dataset (bool, optional): If True, disables augmentations. Default is False.
    
    Returns:
        torchvision.transforms.Compose: A composed set of transformations.
    """
    transforms = []
    if config.get("elastic", False):
        transforms.append(T.RandomAffine(degrees=0, shear=10))
    if config.get("noise", False):
        transforms.append(T.GaussianBlur(kernel_size=3))
    if config.get("dilation_erosion", False):
        transforms.append(T.RandomInvert(p=0.5))
    if config.get("translation", False):
        transforms.append(T.RandomAffine(degrees=0, translate=(0.05, 0.05)))
    
    if val_dataset:
        transforms = []  # No augmentations for validation
    
    transforms.append(T.Resize((32, 128)))  # Standardize input size
    transforms.append(T.ToTensor())  # Convert to tensor
    
    return T.Compose(transforms)


def get_data_loaders(config, save_dir=None):
    """
    Creates and returns training and validation data loaders.

    Args:
        config (dict): Configuration dictionary containing run settings.
        save_dir (str): Directory to save visualization samples (if enabled).
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    data_config = config['data']
    csv_path, train_root, val_root = data_config['csv_path'], data_config['train_root'], data_config['val_root']
    to_visualize_data, vis_mask =  data_config.get('to_visualize_data', False), data_config.get('vis_mask', False)
    batch_size = config['hyperparameters']['batch_size']

    train_transform = get_transforms(config['data_transforms'])
    val_transform = get_transforms(config['data_transforms'], val_dataset=True)

    train_dataset = CaptchaDataset(csv_path, train_root, transform=train_transform)
    val_dataset = CaptchaDataset(csv_path, val_root, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if to_visualize_data and hasattr(utils, 'visualize_data'):
        utils.visualize_data(val_dataset, save_dir, vis_mask=vis_mask)

    return train_loader, val_loader


if __name__=="__main__":
    # dry run to save and visualize the data
    config = {
        "data": {'csv_path': "dataset/captcha_data.csv", 
                 'val_root': "dataset/validation-images/validation-images/",
                 "train_root": "dataset/train-images/train-images/", 
                 "to_visualize_data":True,
                 "vis_mask":True},
        "data_transforms":{
            "elastic": True,
            "noise": True,
            "dilation_erosion": True,
            "translation": True
        },
        "hyperparameters":{
            "batch_size": 32,
        }
    }
    get_data_loaders(config, "./output")