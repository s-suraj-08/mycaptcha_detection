import matplotlib.pyplot as plt
import os
import random
import torch
from torchvision.utils import make_grid
import torchvision.transforms as T

def visualize_data(dataset, save_dir, num_images=16, vis_mask=False):
    """
    Visualize a grid of CAPTCHA images with their ground truth labels.
    
    :param dataset: CaptchaDataset instance
    :param num_images: Number of images to visualize (default 16, 4x4 grid)
    :param vis_mask: Whether to visualize masks instead of original images
    """
    # Ensure we don't try to sample more images than available
    num_images = min(num_images, len(dataset))
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect images and labels
    images = []
    labels = []

    # Get random images from the dataset
    random_ids = random.sample(range(len(dataset)), num_images)
    
    # Sample images
    for random_id in random_ids:
        batch = dataset[random_id]
        label = batch['label_str']
        img = batch['mask'] if vis_mask else batch['image']
        
        # If img is a tensor, convert back to PIL image for display
        if isinstance(img, torch.Tensor):
            # No need to denormalize since current transforms don't apply normalization
            # Just clamp to ensure values are between 0 and 1 for proper display
            img = img.clamp(0, 1)
            img = T.ToPILImage()(img)
        
        images.append(T.ToTensor()(img))
        labels.append(label)
    
    # Create image grid
    grid = make_grid(images, nrow=4, padding=2)
    
    # Display the grid
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis('off')
    
    # Add labels
    plt.title('CAPTCHA Images with Ground Truth Labels')
    
    # Annotate images with their labels
    for i, label in enumerate(labels):
        row = i // 4
        col = i % 4
        plt.text(col * (grid.shape[2] // 4) + 10, 
                 row * (grid.shape[1] // 4) + grid.shape[1] // 4 - 20, 
                 str(label), 
                 color='red', 
                 fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'data_vis.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot metrics
def plot_metrics(plot_file, epochs_list, train_losses, val_losses, train_accs=None, val_accs=None):
    '''
    Function to plot the metrics

    plot_file: save location for the plot
    The remaining are lists of the respective name
    '''
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_losses, '-o', label='Training Loss')
    plt.plot(epochs_list, val_losses, '-o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies if available
    if train_accs is not None and val_accs is not None:
        plt.subplot(1, 2, 2)
        plt.plot(epochs_list, train_accs, '-o', label='Training Accuracy')
        plt.plot(epochs_list, val_accs, '-o', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


# Function to log metrics to file
def log_metrics(log_file, epoch, train_loss, val_loss, train_acc=None, val_acc=None):
    '''
    Function to save metrics as a txt file
    '''
    with open(log_file, 'a') as f:
        log_line = f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        if train_acc is not None and val_acc is not None:
            log_line += f", Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        f.write(log_line + '\n')
    print(log_line)


# Function to save model checkpoint
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc, save_freq, checkpoint_dir, best_model_path, is_best=False):
    '''
    Function to save model checkpoint along with other information
    '''
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    
    # Save regular checkpoint if epoch is multiple of save_freq
    if epoch % save_freq == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")
    
    # Save best model if current model is best
    if is_best:
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved at epoch {epoch}")