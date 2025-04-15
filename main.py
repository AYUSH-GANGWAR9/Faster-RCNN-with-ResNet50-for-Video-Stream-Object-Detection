import os
import torch
import argparse
from dataset import create_data_loaders, resize_dataset_for_memory_efficiency
from model import create_frcnn_resnet50_model
from train import train_model, train_model_with_optimization
from eval import process_video_stream, evaluate_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FRCNN-ResNet50 for Video Stream Object Detection')
    parser.add_argument('--dataset_path', type=str, default=r'C:\Users\gangw\Desktop\Faster RCNN for video object detection\data\davis', help='Path to DAVIS dataset')
    parser.add_argument('--resized_dataset_path', type=str, default=r'C:\Users\gangw\Desktop\Faster RCNN for video object detection\data\davis_resized', help='Path to save resized dataset')
    parser.add_argument('--resize_dataset', action='store_true', help='Resize dataset for memory efficiency')
    parser.add_argument('--target_size', type=tuple, default=(480, 270), help='Target size for resizing (width, height)')
    parser.add_argument('--model_save_path', type=str, default='./model_checkpoints', help='Path to save model checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to load model checkpoint')
    parser.add_argument('--video_path', type=str, default=None, help='Path to test video')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save output video')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'inference'], default='train', 
                        help='Mode to run: train, evaluate, or inference')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories if they don't exist
    os.makedirs(args.model_save_path, exist_ok=True)
    
    # Resize dataset if specified
    if args.resize_dataset:
        print(f"Resizing dataset from {args.dataset_path} to {args.resized_dataset_path}")
        resize_dataset_for_memory_efficiency(args.dataset_path, args.resized_dataset_path, 
                                            target_size=args.target_size)
        dataset_path = args.resized_dataset_path
    else:
        dataset_path = args.dataset_path
    
    # Create model
    model = create_frcnn_resnet50_model(num_classes=2)  # Background + Foreground
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        if os.path.exists(args.load_checkpoint):
            print(f"Loading checkpoint from {args.load_checkpoint}")
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"Checkpoint {args.load_checkpoint} not found.")
    
    # Move model to device
    model.to(device)
    
    if args.mode == 'train':
        # Create data loaders
        print(f"Creating data loaders from {dataset_path}")
        train_loader, val_loader = create_data_loaders(dataset_path, args.batch_size)
        
        # Train model
        print("Starting training...")
        if args.accumulation_steps > 1:
            print(f"Using gradient accumulation with {args.accumulation_steps} steps")
            model = train_model_with_optimization(model, train_loader, val_loader, device, 
                                                args.num_epochs, args.learning_rate, 
                                                args.accumulation_steps)
        else:
            model = train_model(model, train_loader, val_loader, device, 
                                args.num_epochs, args.learning_rate)
        
        # Save final model
        final_model_path = os.path.join(args.model_save_path, "frcnn_resnet50_final.pth")
        print(f"Saving final model to {final_model_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
        }, final_model_path)
    
    elif args.mode == 'evaluate':
        # Create data loaders
        print(f"Creating data loaders from {dataset_path}")
        _, val_loader = create_data_loaders(dataset_path, args.batch_size)
        
        # Evaluate model
        print("Evaluating model...")
        f_measure, s_measure, mae = evaluate_model(model, val_loader, device, 0)
        print(f"Final Results:")
        print(f"F-Measure: {f_measure:.4f}")
        print(f"S-Measure: {s_measure:.4f}")
        print(f"MAE: {mae:.4f}")
    
    elif args.mode == 'inference':
        if args.video_path:
            print(f"Processing video: {args.video_path}")
            process_video_stream(model, args.video_path, args.output_path, device)
        else:
            print("Please provide a video path for inference mode.")
    
    print("Done!")

if __name__ == "__main__":
    main()