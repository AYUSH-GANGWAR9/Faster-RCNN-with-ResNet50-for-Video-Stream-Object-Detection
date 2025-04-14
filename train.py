import torch
from tqdm import tqdm
from eval import evaluate_model, calculate_f_measure, calculate_s_measure, calculate_mae

def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001):
    """
    Train the Faster R-CNN model
    """
    # Move model to the device
    model.to(device)
    
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        
        # Initialize metrics
        running_loss = 0.0
        running_loss_classifier = 0.0
        running_loss_box_reg = 0.0
        running_loss_objectness = 0.0
        running_loss_rpn_box_reg = 0.0
        
        # Process each batch
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move images and targets to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize
            losses.backward()
            optimizer.step()
            
            # Update running losses
            running_loss += losses.item()
            running_loss_classifier += loss_dict['loss_classifier'].item()
            running_loss_box_reg += loss_dict['loss_box_reg'].item()
            running_loss_objectness += loss_dict['loss_objectness'].item()
            running_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        
        # Update learning rate
        lr_scheduler.step()
        
        # Calculate average losses
        avg_loss = running_loss / len(train_loader)
        avg_loss_classifier = running_loss_classifier / len(train_loader)
        avg_loss_box_reg = running_loss_box_reg / len(train_loader)
        avg_loss_objectness = running_loss_objectness / len(train_loader)
        avg_loss_rpn_box_reg = running_loss_rpn_box_reg / len(train_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Classifier Loss: {avg_loss_classifier:.4f}")
        print(f"Box Reg Loss: {avg_loss_box_reg:.4f}")
        print(f"Objectness Loss: {avg_loss_objectness:.4f}")
        print(f"RPN Box Reg Loss: {avg_loss_rpn_box_reg:.4f}")
        
        # Validation
        evaluate_model(model, val_loader, device, epoch)
    
    return model

def train_model_with_optimization(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001, accumulation_steps=4):
    """
    Train the Faster R-CNN model with gradient accumulation for memory optimization
    """
    # Move model to the device
    model.to(device)
    
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        
        # Initialize metrics
        running_loss = 0.0
        running_loss_classifier = 0.0
        running_loss_box_reg = 0.0
        running_loss_objectness = 0.0
        running_loss_rpn_box_reg = 0.0
        
        # Process each batch
        for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move images and targets to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            # Normalize the loss to account for accumulation
            losses = losses / accumulation_steps
            
            # Backward pass
            losses.backward()
            
            # Update running losses
            running_loss += losses.item() * accumulation_steps
            running_loss_classifier += loss_dict['loss_classifier'].item() / accumulation_steps
            running_loss_box_reg += loss_dict['loss_box_reg'].item() / accumulation_steps
            running_loss_objectness += loss_dict['loss_objectness'].item() / accumulation_steps
            running_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item() / accumulation_steps
            
            # Update weights if we've accumulated enough gradients
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
        
        # Update learning rate
        lr_scheduler.step()
        
        # Calculate average losses
        avg_loss = running_loss / len(train_loader)
        avg_loss_classifier = running_loss_classifier / len(train_loader)
        avg_loss_box_reg = running_loss_box_reg / len(train_loader)
        avg_loss_objectness = running_loss_objectness / len(train_loader)
        avg_loss_rpn_box_reg = running_loss_rpn_box_reg / len(train_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Classifier Loss: {avg_loss_classifier:.4f}")
        print(f"Box Reg Loss: {avg_loss_box_reg:.4f}")
        print(f"Objectness Loss: {avg_loss_objectness:.4f}")
        print(f"RPN Box Reg Loss: {avg_loss_rpn_box_reg:.4f}")
        
        # Validation
        evaluate_model(model, val_loader, device, epoch)
    
    return model