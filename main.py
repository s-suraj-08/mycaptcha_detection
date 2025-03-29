import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_data_loaders
from model import get_cnn_model, CRNN
import utils
import yaml
import os
import numpy as np
from datetime import datetime
import shutil
import matplotlib.pyplot as plt

CONFIG_PATH = 'config.yaml'

def load_config_yaml(config_file_path):
    with open(config_file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def calc_captcha_accuracy(predictions, label_strs):
    """
    Calculate accuracy for multi-digit captcha predictions

    Args:
        predictions: List of lists of digits
        label_strs: List of ground truth label strings
        
    Returns:
        float: Accuracy (percentage of completely correct captchas)
    """
    correct = 0
    total = len(predictions)
    
    for pred_digits, true_label in zip(predictions, label_strs):
        # Convert list of predicted digits into a single string for comparison
        pred_str = ''.join([str(d) for d in pred_digits])
        
        # Check if prediction matches ground truth
        if pred_str == true_label:
            correct += 1
    
    return correct / total if total > 0 else 0

def main():
    config = load_config_yaml(CONFIG_PATH)

    # Create run directory
    run_name = config['general']['run_name']
    run_dir = os.path.join('output', run_name)

    # Ensure run directory exists
    if os.path.exists(run_dir):
        print(f"Error: Run directory {run_dir} already exists, change run name")
        return
    os.makedirs(run_dir, exist_ok=True)

    # Setup paths for logging and saving
    log_file = os.path.join(run_dir, 'training_log.txt')
    plot_file = os.path.join(run_dir, 'training_plot.png')
    config_backup = os.path.join(run_dir, 'config_backup.yaml')

    # Copy config file to run directory for reproducibility
    shutil.copy2(CONFIG_PATH, config_backup)

    # Save frequency for model checkpoints
    save_freq = config['general'].get('save_freq', 5)  # Save model every 5 epochs by default
    best_model_path = os.path.join(run_dir, 'best_model.pth')
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load training and validation data with necessary augmentations/transforms
    train_loader, val_loader = get_data_loaders(config, run_dir)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    if config['hyperparameters']['which_model'] == 'crnn':
        model = CRNN(num_classes=10).to(device)
    elif config['hyperparameters']['which_model'] == 'cnn':
        model = get_cnn_model().to(device)
    else:
        raise ValueError(f"Unknown model type: {config['hyperparameters']['which_model']}")

    # Optimizer & Loss
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['hyperparameters']['lr'],
        weight_decay=config['hyperparameters'].get('weight_decay', 0)
    )

    # Loss function
    if config['hyperparameters']['which_loss'] == 'ctc':
        criterion = nn.CTCLoss(blank=10, reduction='mean', zero_infinity=True)
    elif config['hyperparameters']['which_loss'] == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {config['hyperparameters']['which_loss']}")

    # Training loop
    epochs = config['hyperparameters'].get('epochs', 10)
    
    # Lists to store metrics for plotting
    epochs_list = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Write header to log file
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run directory: {run_dir}\n")
        f.write(f"Model: {type(model).__name__}\n")
        f.write(f"Optimizer: {type(optimizer).__name__}, lr={optimizer.param_groups[0]['lr']}\n")
        f.write(f"Loss: {type(criterion).__name__}\n")
        f.write(f"Device: {device}\n")
        f.write("=" * 50 + "\n")

    best_val_loss = float('inf')
    
    # Train for specified number of epochs
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch['mask'].to(device)  # Use preprocessed mask instead of raw image to reduce noise in training
            labels = batch['label'].to(device)
            label_lengths = batch['label_length'].to(device)
            label_strs = batch['label_str']  # Keep original string labels for accuracy
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            if config['hyperparameters']['which_loss'] == 'ctc':
                # For CTC loss, we need to handle sequences
                batch_size, seq_len, _ = outputs.size()
                
                # Input lengths is the sequence length for each batch item
                # CTC loss requires specifying input sequence lengths for alignment
                input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
                
                # Compute CTC loss
                loss = criterion(outputs.transpose(0, 1), labels, input_lengths, label_lengths)
            else:
                # For CrossEntropyLoss (CNN model)
                # Reshape output to match labels
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Decode predictions for accuracy calculation
            if config['hyperparameters']['which_model'] == 'crnn':
                predictions = model.decode_prediction(outputs)
                batch_acc = calc_captcha_accuracy(predictions, label_strs)
                train_correct += batch_acc * batch_size
                train_total += batch_size
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Calculate average training metrics
        train_loss_avg = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                images = batch['mask'].to(device)
                labels = batch['label'].to(device)
                label_lengths = batch['label_length'].to(device)
                label_strs = batch['label_str']
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                if config['hyperparameters']['which_loss'] == 'ctc':
                    batch_size, seq_len, _ = outputs.size()
                    input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
                    loss = criterion(outputs.transpose(0, 1), labels, input_lengths, label_lengths)
                else:
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                val_loss += loss.item()
                
                # Calculate accuracy
                if config['hyperparameters']['which_model'] == 'crnn':
                    predictions = model.decode_prediction(outputs)
                    batch_acc = calc_captcha_accuracy(predictions, label_strs)
                    val_correct += batch_acc * len(labels)
                    val_total += len(labels)
        
        # Calculate average validation metrics
        val_loss_avg = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Log metrics
        utils.log_metrics(log_file, epoch+1, train_loss_avg, val_loss_avg, train_acc, val_acc)
        
        # Store metrics for plotting
        epochs_list.append(epoch+1)
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update plot
        utils.plot_metrics(plot_file, epochs_list, train_losses, val_losses, train_accs, val_accs)
        
        # Check if current model is the best
        is_best = val_loss_avg < best_val_loss
        if is_best:
            best_val_loss = val_loss_avg
        
        # Save checkpoint
        utils.save_checkpoint(
            model, optimizer, epoch+1, 
            train_loss_avg, val_loss_avg, 
            train_acc, val_acc, 
            save_freq, checkpoint_dir, best_model_path,
            is_best
        )
    
    # Log final results
    with open(log_file, 'a') as f:
        f.write("=" * 50 + "\n")
        f.write(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Final Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}\n")
        f.write(f"Final Train Acc: {train_accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")

    print(f"Training complete. Results saved to {run_dir}")

if __name__ == "__main__":
    main()