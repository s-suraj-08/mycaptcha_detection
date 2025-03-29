from data import get_data_loaders
import os
from main import load_config_yaml
from model import get_cnn_model, CRNN
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

def calc_captcha_metrics(predictions, label_strs):
    """
    Calculate accuracy, precision, and recall for multi-digit captcha predictions.

    Args:
        predictions: List of lists of digits (predicted values)
        label_strs: List of ground truth label strings
        
    Returns:
        Accuracy on the captchas, predicted charaters(as int), gt character (as int)
    """
    correct = 0
    total = len(predictions)
    all_preds = []
    all_labels = []
    
    for pred_digits, true_label in zip(predictions, label_strs):
        # Convert list of predicted digits into a single string
        pred_str = ''.join(map(str, pred_digits))
        
        # Check if the entire prediction is correct
        if pred_str == true_label:
            correct += 1
        
        # maintain 6 character precution length
        if len(pred_str)>6:
            pred_str=pred_str[:6]
        if len(pred_str)<6:
            pred_str = f"{pred_str:06s}"
        
        # Collect individual character predictions and labels for precision/recall
        all_preds.extend([int(digit) for digit in pred_str])
        all_labels.extend([int(digit) for digit in true_label])
    
    accuracy = correct / total if total > 0 else 0

    return accuracy, all_preds, all_labels

def load_model(config, best_model_path, device):
    # Get the model
    if config['hyperparameters']['which_model'] == 'crnn':
        model = CRNN(num_classes=10).to(device)
    elif config['hyperparameters']['which_model'] == 'cnn':
        model = get_cnn_model().to(device)
    else:
        raise ValueError("Unknown model type")
    
    # Load best model
    checkpnt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpnt['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, data_loader, device):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        count=0
        for batch in data_loader:
            images = batch['mask'].to(device)
            label_strs = batch['label_str']
            
            outputs = model(images)
            preds = model.decode_prediction(outputs)
            
            all_preds.extend(preds)
            all_labels.extend(label_strs)

            if count%10==0:
                print(f"Done with {count}/{len(data_loader)}")
            count+=1

        # get the accuracy for the captcha
        accuracy_captcha, char_all_preds, char_all_labels = calc_captcha_metrics(all_preds, all_labels)

        # get the character level accuracy, precision, recall
        char_accuracy = accuracy_score(char_all_labels, char_all_preds)
        char_precision = precision_score(char_all_labels, char_all_preds, average='macro', zero_division=0)
        char_recall = recall_score(char_all_labels, char_all_preds, average='macro', zero_division=0)

        # get the confusion matrix
        unique_chars = [i for i in range(10)]
        cm = confusion_matrix(char_all_preds, char_all_labels, labels=unique_chars)
        return accuracy_captcha, char_accuracy, char_precision, char_recall, cm


def plot_confusion_matrix(cm, title, savepath):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(savepath)

def inference(config):
    # Get the data
    train_loader, val_loader = get_data_loaders(config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Best model path
    best_model_path = os.path.join('output', config['general']['run_name'], 'best_model.pth')
    
    # Load model
    model = load_model(config, best_model_path, device)
    
    # Evaluate on train and validation sets
    train_acc_capthca, train_acc, train_prec, train_rec, train_cm = evaluate_model(model, train_loader, device)
    val_acc_captcha, val_acc, val_prec, val_rec, val_cm = evaluate_model(model, val_loader, device)
    
    # Print the statistics
    print(f"Train Accuracy of detecting captcha: {train_acc_capthca:.4f}")
    print(f"Validation Accuracy of detecting captcha: {val_acc_captcha:.4f}")
    print('-'*100)
    print("Character level statistics below")
    print(f"Train Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")
    
    # Plot & save the confusion matrices
    plot_confusion_matrix(train_cm, "Train Set Confusion Matrix", "train_confusion_matrix.png")
    plot_confusion_matrix(val_cm, "Validation Set Confusion Matrix", "validation_confusion_matrix.png")

if __name__ == "__main__":
    rundir = './output/best_run'
    config_file_path = os.path.join(rundir, 'config_backup.yaml')
    config = load_config_yaml(config_file_path)

    # Override
    config['data']['to_visualize_data'] = False

    inference(config)
