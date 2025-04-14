import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm  

class DAVISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with the DAVIS dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get list of all videos
        self.videos = [d for d in os.listdir(os.path.join(root_dir, 'JPEGImages', '480p')) 
                      if os.path.isdir(os.path.join(root_dir, 'JPEGImages', '480p', d))]
        
        # Create a list of all frame paths and corresponding annotations
        self.frames = []
        self.annotations = []
        
        for video in self.videos:
            video_frames = sorted(os.listdir(os.path.join(root_dir, 'JPEGImages', '480p', video)))
            video_annots = sorted(os.listdir(os.path.join(root_dir, 'Annotations', '480p', video)))
            
            for frame, annot in zip(video_frames, video_annots):
                self.frames.append(os.path.join(root_dir, 'JPEGImages', '480p', video, frame))
                self.annotations.append(os.path.join(root_dir, 'Annotations', '480p', video, annot))
                
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        annot_path = self.annotations[idx]
        
        # Read image and annotation
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE)
        # Convert mask to binary
        mask = (mask > 0).astype(np.uint8)
        
        # Create bounding boxes from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x+w, y+h])
        
        # If no contours found, create a dummy box
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]  # Dummy box
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.ones((len(boxes),), dtype=torch.int64)  # All boxes have class 1 (foreground)
        target["masks"] = torch.as_tensor(mask, dtype=torch.uint8).unsqueeze(0)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

# Data transformations
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def download_davis_dataset(root_dir):
    """
    Download the DAVIS dataset if not already downloaded
    """
    # Check if dataset already exists
    if os.path.exists(os.path.join(root_dir, 'JPEGImages')) and os.path.exists(os.path.join(root_dir, 'Annotations')):
        print("DAVIS dataset already exists.")
        return
    
    print("Please download the DAVIS dataset manually from https://davischallenge.org/davis2017/code.html")
    print("and extract it to", root_dir)

def resize_dataset_for_memory_efficiency(source_dir, target_dir, target_size=(480, 270)):
    """
    Resize the DAVIS dataset for memory efficiency
    """
    # Create target directories if they don't exist
    os.makedirs(os.path.join(target_dir, 'JPEGImages', '480p'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'Annotations', '480p'), exist_ok=True)
    
    # Get list of all videos
    videos = [d for d in os.listdir(os.path.join(source_dir, 'JPEGImages', '480p')) 
              if os.path.isdir(os.path.join(source_dir, 'JPEGImages', '480p', d))]
    
    # Process each video
    for video in videos:
        # Create video directories if they don't exist
        os.makedirs(os.path.join(target_dir, 'JPEGImages', '480p', video), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'Annotations', '480p', video), exist_ok=True)
        
        # Process frames
        frames = sorted(os.listdir(os.path.join(source_dir, 'JPEGImages', '480p', video)))
        for frame in tqdm(frames, desc=f"Resizing frames for {video}"):
            # Read and resize frame
            img_path = os.path.join(source_dir, 'JPEGImages', '480p', video, frame)
            img = cv2.imread(img_path)
            resized_img = cv2.resize(img, target_size)
            
            # Save resized frame
            cv2.imwrite(os.path.join(target_dir, 'JPEGImages', '480p', video, frame), resized_img)
        
        # Process annotations
        annotations = sorted(os.listdir(os.path.join(source_dir, 'Annotations', '480p', video)))
        for annot in tqdm(annotations, desc=f"Resizing annotations for {video}"):
            # Read and resize annotation
            annot_path = os.path.join(source_dir, 'Annotations', '480p', video, annot)
            mask = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE)
            resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            
            # Save resized annotation
            cv2.imwrite(os.path.join(target_dir, 'Annotations', '480p', video, annot), resized_mask)
    
    print(f"Dataset resized to {target_size} and saved to {target_dir}")

def create_data_loaders(dataset_path, batch_size=2):
    """
    Create data loaders for training and validation
    """
    # Download dataset if needed
    download_davis_dataset(dataset_path)
    
    # Create dataset
    dataset = DAVISDataset(root_dir=dataset_path, transform=get_transform())
    
    # Split dataset into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    return train_loader, val_loader

def collate_fn(batch):
    return tuple(zip(*batch))
