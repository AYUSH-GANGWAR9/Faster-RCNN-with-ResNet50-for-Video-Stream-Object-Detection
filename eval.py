import torch
import numpy as np
import cv2
from tqdm import tqdm
from dataset import get_transform

def calculate_f_measure(pred_mask, gt_mask, beta=0.3):
    """
    Calculate F-Measure as described in the paper
    """
    # Calculate precision and recall
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    precision = intersection / (pred_mask.sum() + 1e-8)
    recall = intersection / (gt_mask.sum() + 1e-8)
    
    # Calculate F-Measure
    f_measure = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
    
    return f_measure

def calculate_s_measure(pred_mask, gt_mask, alpha=0.4):
    """
    Calculate S-Measure (Structure measure) as described in the paper
    """
    # Calculate object-aware structural similarity (So)
    # This is a simplified implementation
    pred_fg = pred_mask == 1
    gt_fg = gt_mask == 1
    
    # Calculate region similarity
    intersection_fg = np.logical_and(pred_fg, gt_fg).sum()
    so = (2 * intersection_fg) / (pred_fg.sum() + gt_fg.sum() + 1e-8)
    
    # Calculate region-aware structural similarity (Sr)
    pred_bg = pred_mask == 0
    gt_bg = gt_mask == 0
    
    intersection_bg = np.logical_and(pred_bg, gt_bg).sum()
    sr = (2 * intersection_bg) / (pred_bg.sum() + gt_bg.sum() + 1e-8)
    
    # Combine object and region awareness
    s_measure = alpha * so + (1 - alpha) * sr
    
    return s_measure

def calculate_mae(pred_mask, gt_mask):
    """
    Calculate Mean Absolute Error as described in the paper
    """
    # Normalize masks
    pred_mask = pred_mask.astype(np.float32) / 255.0 if pred_mask.max() > 1 else pred_mask.astype(np.float32)
    gt_mask = gt_mask.astype(np.float32) / 255.0 if gt_mask.max() > 1 else gt_mask.astype(np.float32)
    
    # Calculate MAE
    mae = np.abs(pred_mask - gt_mask).mean()
    
    return mae

def evaluate_model(model, val_loader, device, epoch):
    """
    Evaluate the model on the validation set
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics for evaluation
    running_f_measure = 0.0
    running_s_measure = 0.0
    running_mae = 0.0
    
    # Process each batch
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            # Move images to device
            images = list(image.to(device) for image in images)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate metrics
            for i, output in enumerate(outputs):
                # Get ground truth mask
                gt_mask = targets[i]['masks'].squeeze().cpu().numpy()
                
                # Get predicted masks
                pred_masks = output['masks'].squeeze().cpu().numpy()
                
                # If there are multiple masks, combine them
                if len(pred_masks.shape) > 2:
                    pred_mask = np.max(pred_masks, axis=0)
                else:
                    pred_mask = pred_masks
                
                # Threshold the mask
                pred_mask = (pred_mask > 0.5).astype(np.uint8)
                
                # Calculate metrics
                f_measure = calculate_f_measure(pred_mask, gt_mask)
                s_measure = calculate_s_measure(pred_mask, gt_mask)
                mae = calculate_mae(pred_mask, gt_mask)
                
                running_f_measure += f_measure
                running_s_measure += s_measure
                running_mae += mae
    
    # Calculate average metrics
    avg_f_measure = running_f_measure / len(val_loader.dataset)
    avg_s_measure = running_s_measure / len(val_loader.dataset)
    avg_mae = running_mae / len(val_loader.dataset)
    
    print(f"Validation Results:")
    print(f"F-Measure: {avg_f_measure:.4f}")
    print(f"S-Measure: {avg_s_measure:.4f}")
    print(f"MAE: {avg_mae:.4f}")
    
    return avg_f_measure, avg_s_measure, avg_mae

def process_video_stream(model, video_path, output_path=None, device='cpu', confidence_threshold=0.5):
    """
    Process a video stream and detect objects
    """
    # Load the model to the specified device
    model.to(device)
    model.eval()
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer if output path is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transform
            transform = get_transform()
            image = transform(rgb_frame).unsqueeze(0).to(device)
            
            # Get predictions
            prediction = model(image)
            
            # Process prediction
            boxes = prediction[0]['boxes'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()
            masks = prediction[0]['masks'].cpu().numpy()
            
            # Filter predictions by confidence threshold
            indices = np.where(scores > confidence_threshold)[0]
            
            # Draw boxes and masks on the frame
            for i in indices:
                box = boxes[i].astype(int)
                mask = masks[i, 0] > 0.5
                
                # Draw box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Apply mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask] = [0, 0, 255]  # Red mask
                frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)
                
                # Add score
                cv2.putText(frame, f"{scores[i]:.2f}", (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Video Stream Object Detection', frame)
            
            # Write frame to output video if specified
            if output_path:
                out.write(frame)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()