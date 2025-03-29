import torch
import torch.nn as nn
import torchvision.models as models

# CRNN Model
class CRNN(nn.Module):
    def __init__(self, num_classes=10, hidden_size=256):
        """
        CRNN (Convolutional Recurrent Neural Network) for sequence recognition.
        
        Args:
            num_classes (int): Number of classes (10 for digits 0-9, plus 1 for CTC blank)
            hidden_size (int): Size of LSTM hidden state
        """
        super(CRNN, self).__init__()
        
        # CNN layers for feature extraction
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x128 -> 16x64
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x64 -> 8x32
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8x32 -> 4x32
            
            # Layer 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Layer 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4x32 -> 2x32
        )
        
        # Bidirectional LSTM
        self.rnn = nn.Sequential(
            nn.LSTM(512 * 2, hidden_size, bidirectional=True, batch_first=True),
            nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        )
        
        # Final classifier
        self.num_classes = num_classes
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)  # +1 for CTC blank
        
    def forward(self, x):
        # CNN feature extraction
        conv = self.cnn(x)  # [batch_size, channels, height, width]
        
        # Reshape for RNN
        batch_size, channels, height, width = conv.size()
        conv = conv.permute(0, 3, 1, 2)  # [batch_size, width, channels, height]
        conv = conv.reshape(batch_size, width, channels * height)  # [batch_size, width, channels*height]
        
        # RNN sequence processing
        rnn_input = conv
        self.rnn_output, _ = self.rnn[0](rnn_input)
        self.rnn_output, _ = self.rnn[1](self.rnn_output)
        
        # Final prediction
        output = self.fc(self.rnn_output)  # [batch_size, seq_len, num_classes+1]
        
        # For CTC loss, we need log softmax along the class dimension
        output = nn.functional.log_softmax(output, dim=2)
        return output

    def decode_prediction(self, pred, blank_label=10):
        """
        Decode the model's prediction using CTC decoding.
        
        Args:
            pred (Tensor): Model prediction with shape [batch_size, seq_len, num_classes]
            blank_label (int): The index of the blank label
            
        Returns:
            list: List of decoded sequences (indices)
        """
        # Get the best prediction for each timestep
        pred = pred.argmax(dim=2)  # [batch_size, seq_len]
        
        batch_size = pred.size(0)
        decoded = []
        for b in range(batch_size):
            seq = []
            prev = -1
            # Simple greedy decoding
            for t in range(pred.size(1)):
                label = pred[b, t].item()
                
                # CTC rule: remove repeated labels and blanks
                if label != blank_label and label != prev:
                    seq.append(label)
                prev = label
            
            decoded.append(seq)
        return decoded

# CNN Model (Alternative approach)
def get_cnn_model():
    model = models.resnet18(pretrained=True)
    # Convert first layer to accept grayscale
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Adjust final layer for 6-digit CAPTCHA with 10 possible classes per digit
    model.fc = nn.Linear(model.fc.in_features, 6 * 10)
    return model