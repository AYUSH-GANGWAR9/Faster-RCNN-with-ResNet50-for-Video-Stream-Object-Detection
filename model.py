import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads

def create_frcnn_resnet50_model(num_classes=2):  # Background + Foreground
    """
    Create Faster R-CNN model with ResNet50 backbone as described in the paper
    with added mask prediction capabilities
    """
    # Load pre-trained ResNet50 backbone
    backbone = torchvision.models.resnet50(pretrained=True)
    
    # Remove the last two layers (avgpool and fc)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    
    # Define the feature extractor
    backbone.out_channels = 2048  # ResNet50 output channels
    
    # RPN parameters
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),  # Different sizes to handle scale variations
        aspect_ratios=((0.5, 1.0, 2.0),)   # Different aspect ratios to handle shape variations
    )
    
    # ROI Pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],  # Use the output from the backbone
        output_size=7,        # ROI Pooling output size
        sampling_ratio=2      # Sampling ratio
    )
    
    # Create Faster R-CNN model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    # Replace the box predictor with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Add mask prediction capabilities to the model
    
    # First, create a mask pooler
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=14,
        sampling_ratio=2)
    
    # Create a mask predictor
    mask_predictor = MaskRCNNPredictor(
        in_channels=in_features,
        dim_reduced=256,
        num_classes=num_classes)
    
    # Add the mask pooler and predictor to the model
    model.roi_heads.mask_roi_pool = mask_roi_pooler
    model.roi_heads.mask_predictor = mask_predictor
    
    return model