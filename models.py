import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as pt_models
class ANNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ANNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)   # first hidden layer
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)          # second hidden layer
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, num_classes)  # output layer

    def forward(self, x):
        # Flatten image -> (batch_size, input_size)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier,self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class MultiClassClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class CNNClassifier(nn.Module):
    def __init__(self,num_classes):
        super(CNNClassifier,self).__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),#128 x 128-->64 x 64
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),#64 x 64-->32 x 32
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),#32 x 32 -->16 x 16
        )
        self.fc_layers=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16,256),
            nn.ReLU(),
            nn.Linear(256,num_classes)
        )
    def forward(self,x):
        return self.fc_layers(self.conv_layers(x))

    def visualize_layer_outputs(model, input_image, layers_to_visualize=None, max_features=16):
        model.eval()
        x = input_image.to(next(model.parameters()).device)

        if layers_to_visualize is None:
            layers_to_visualize = [0, 3, 6]

        with torch.no_grad():
            for idx, layer in enumerate(model.conv_layers):
                x = layer(x)
                if idx in layers_to_visualize:
                    num_features = min(x.shape[1], max_features)
                    plt.figure(figsize=(12, 6))
                    for i in range(num_features):
                        plt.subplot(4, 4, i + 1)
                        plt.imshow(x[0, i].cpu(), cmap='viridis')
                        plt.axis('off')
                    plt.suptitle(f'Layer {idx} Feature Maps')
                    plt.show()
class CNNClassifier_regularization(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNClassifier_regularization, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def get_model(task_type,num_classes=None):
    """Returns the model based on classification type."""
    if task_type == "binary":
        return BinaryClassifier()
    elif task_type == "multi":
        if num_classes is None:
            raise ValueError("num_classes must be specified for multi-class classification")
        return MultiClassClassifier(num_classes)
    else:
        raise ValueError("Invalid task type. Choose 'binary' or 'multi'.")



def get_optimizer(optimizer_name,model,lr=0.01):
    if optimizer_name=="SGD":
        return optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    elif optimizer_name=="Adam":
        return optim.Adam(model.parameters(),lr=lr)
    elif optimizer_name=="RMSprop":
        return optim.RMSprop(model.parameters(),lr=lr)
    else:
        raise ValueError("Invalid optimizer name.Choose from 'SGD','Adam','RMSprop'")
class VGGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGClassifier, self).__init__()
        # Load pretrained VGG16
        self.model = pt_models.vgg16(pretrained=True)
        # Freeze feature extractor layers (optional)
        for param in self.model.features.parameters():
            param.requires_grad = False
            # Replace final classifier layer dynamically
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class AlexNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetClassifier, self).__init__()
        # Load pretrained AlexNet
        self.model = pt_models.alexnet(pretrained=True)
        # Freeze feature extractor layers (optional)
        for param in self.model.features.parameters():
            param.requires_grad = False
            # Replace final classifier layer dynamically
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class AutoEncoder(nn.Module):
    def __init__(self, encoded_dim=256):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),   # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 32 -> 16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, encoded_dim)        # ✅ fixed here
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 256 * 16 * 16),       # ✅ match encoder output
            nn.ReLU(),
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),    # 64 -> 128
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded







class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class EncoderClassifierWrapper(nn.Module):
    def __init__(self, autoencoder, classifier):
        super().__init__()
        self.encoder = autoencoder.encoder
        self.classifier = classifier

    def forward(self, x):
        with torch.no_grad():  # freeze encoder if desired
            encoded = self.encoder(x)
        out = self.classifier(encoded)
        return out


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Recurrent Layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidirectional)

        # Fully connected output layer
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_length, input_size)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)
        # Get last time step output
        out = out[:, -1, :]
        out = self.fc(out)
        return out

import torch
import torch.nn as nn





def visualize_layer_outputs(model, input_image, layers_to_visualize=None, max_features=16):

    model.eval()
    x = input_image.to(next(model.parameters()).device)

    if layers_to_visualize is None:
        layers_to_visualize = [0, 3, 6]

    with torch.no_grad():
        for idx, layer in enumerate(model.conv_layers):
            x = layer(x)
            if idx in layers_to_visualize:
                num_features = min(x.shape[1], max_features)
                plt.figure(figsize=(12, 6))
                for i in range(num_features):
                    plt.subplot(4, 4, i + 1)
                    plt.imshow(x[0, i].cpu(), cmap='viridis')
                    plt.axis('off')
                plt.suptitle(f'Layer {idx} Feature Maps')
                plt.show()

def visualize_reconstruction(model, dataloader, device, num_images=6):
    model.eval()
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.to(device)

    with torch.no_grad():
        encoded, outputs = model(images)   # <-- Unpack tuple here

    # Move to CPU
    images = images.cpu()
    outputs = outputs.cpu()

    # De-normalize from [-1, 1] → [0, 1]
    images = (images + 1) / 2
    outputs = (outputs + 1) / 2

    # Plot
    fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
    for i in range(num_images):
        axes[0, i].imshow(images[i].permute(1, 2, 0).squeeze())
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(outputs[i].permute(1, 2, 0).squeeze())
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    plt.show()


def get_model_by_name(name, num_classes=None, **kwargs):
    name = name.lower()
    if name == 'vgg':
        if num_classes is None:
            raise ValueError('num_classes required for vgg')
        return VGGClassifier(num_classes, **kwargs)
    if name == 'resnet':
        if num_classes is None:
            raise ValueError('num_classes required for resnet')
        return ResNetClassifier(num_classes, **kwargs)
    if name == 'mobilenet':
        if num_classes is None:
            raise ValueError('num_classes required for mobilenet')
        return MobileNetClassifier(num_classes, **kwargs)
    if name == 'lstm':
        # use kwargs for input_size etc.
        return LSTMClassifier(kwargs.get('input_size', 128), kwargs.get('hidden_size', 256),
                              kwargs.get('num_layers', 2), kwargs.get('num_classes', 2),
                              kwargs.get('bidirectional', False))
    raise ValueError('unknown model name')


