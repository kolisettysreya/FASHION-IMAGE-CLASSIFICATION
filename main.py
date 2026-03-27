import random
import os
import torch
import torch.nn as nn

def train(model, train_loader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(criterion, nn.BCELoss):
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss / len(train_loader):.4f} Accuracy: {accuracy:.2f}%")


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(criterion, nn.BCELoss):
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            val_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss=val_loss/len(val_loader)
    print(f"Validation Loss: {avg_loss:.2f}, Validation Accuracy: {accuracy:.2f}%")
    return accuracy,avg_loss

def evaluate_random_minibatch(model,val_loader,criterion,device,num_batches=1):
    model.eval()
    val_loss=0.0
    correct=0
    total=0

    all_batches=list(val_loader)
    chosen_batches=random.sample(all_batches,min(num_batches,len(all_batches)))

    with torch.no_grad():
        for images,labels in chosen_batches:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)

            if isinstance(criterion,nn.BCELoss):
                labels=labels.float().unsqueeze(1)
                loss=criterion(outputs,labels)
                predicted=(outputs>0.5).float()
            else:
                loss=criterion(outputs,labels)
                _,predicted=torch.max(outputs,1)

            val_loss+=loss.item()
            correct+=(predicted==labels).sum().item()
            total+=labels.size(0)
        accuracy=100*correct/total
        avg_loss=val_loss/len(val_loader)
        print(f"Validate Loss:{avg_loss:.4f},Validation Accuracy: {accuracy:.4f}")
        return accuracy,avg_loss

import os
import torch
import torch.nn as nn
import torch.optim as optim
import Preprocess as pre
import models
import results as rp
from torchviz import make_dot



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")



    data_dir = r"C:\Endsemlab\DL\DATASET\masterCategory\SportsClassification"
    train_loader, test_loader, validate_loader, classes = pre.load_data(data_dir, batch_size=32, augment=True)

    print("DEBUG: Looking for training data at:", os.path.join(data_dir, "train"))
    print("Exists?", os.path.exists(os.path.join(data_dir, "train")))

    classes = train_loader.dataset.classes
    print(f"\nClasses: {classes}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(validate_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")

    # Optional: visualize some training samples
    pre.visualize_data(train_loader, classes, num_samples=8)





    task_type = "binary" if len(classes) == 2 else "multi"
    num_classes = len(classes) if task_type == "multi" else None


    model = models.get_model(task_type, num_classes=num_classes).to(device)


    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("\n🔹 Training ANN Model...")

    # input_size = channels * height * width
    sample_input, _ = next(iter(train_loader))
    input_size = sample_input[0].numel()  # flatten one image

    ann_model = models.ANNClassifier(input_size=input_size, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ann_model.parameters(), lr=0.001)

    train(ann_model, train_loader, optimizer, criterion, device, epochs=10)
    evaluate(ann_model, validate_loader, criterion, device)

    print("\n🚀 Training the model...")
    train(model, train_loader, optimizer, criterion, device, epochs=10)
    #
    #
    print("\n📊 Evaluating on validation set...")
    evaluate(model, validate_loader, criterion, device)
    results={}
    for opt_name in {"SGD","Adam","RMSprop"}:
        print(f"\n=====Training with {opt_name} Optimizer=====")
        model=models.get_model(task_type,num_classes=num_classes).to(device)
        optimizer=models.get_optimizer(opt_name,model,lr=0.001)

        print("\n Training the model...")
        train(model,train_loader,optimizer, criterion, device,epochs=10)

        print("\n Evaluating the validation set...")
        acc,loss=evaluate(model,validate_loader,criterion, device)
        results[opt_name] = {"accuracy": acc, "loss": loss}
        rp.plot_optimizer_results(results)

    print("\nEvaluating on random mini batch validation set...")
    evaluate_random_minibatch(model, validate_loader, criterion, device)

    model = models.CNNClassifier(num_classes=num_classes).to(device)
    print("CNN Classifier", model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, optimizer, criterion, device, epochs=10)
    evaluate(model, validate_loader, criterion, device)
    evaluate_random_minibatch(model, validate_loader, criterion, device)

    print("\n Visualizing CNN Layer outputs for a sample image....")
    sample_image, _ = next(iter(validate_loader))
    sample_image = sample_image[0].unsqueeze(0)
    model.visualize_layer_outputs(sample_image)

    # call standalone visualize function


    # Train new CNN with regularization
    model2 = models.CNNClassifier_regularization(num_classes=num_classes).to(device)
    print(model2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=0.001)

    train(model2, train_loader, optimizer, criterion, device, epochs=10)
    evaluate(model2, validate_loader, criterion, device)

    # === Save model + classes ===
    import json
    os.makedirs('saved_models', exist_ok=True)

    torch.save(model2.state_dict(), 'saved_models/cnn_v1.pth')

    classes = train_loader.dataset.classes
    with open('saved_models/classes.json', 'w') as f:
        json.dump(list(classes), f)

    print("Saved model weights -> saved_models/cnn_v1.pth")
    print("Saved classes -> saved_models/classes.json")

    vgg_model = models.VGGClassifier(num_classes=num_classes).to(device)
    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg_model.parameters(), lr=0.001)
    print("VGG16 model training...")
    train(vgg_model, train_loader, optimizer, criterion, device, epochs=10)
    evaluate(vgg_model, validate_loader, criterion, device)

    # AlexNet model
    alex_model = models.AlexNetClassifier(num_classes=num_classes).to(device)
    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(alex_model.parameters(), lr=0.001)
    print("AlexNet Model Training...")
    train(alex_model, train_loader, optimizer, criterion, device, epochs=10)
    evaluate(alex_model, validate_loader, criterion, device)


    autoencoder = models.AutoEncoder(encoded_dim=256)
    autoencoder.eval()  # freeze encoder
    classifier = models.ResNetClassifier(num_classes, encoded_dim=256)
    model = models.EncoderClassifierWrapper(autoencoder, classifier).to(device)

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training Encoder-Classifier Wrapper...")
    train(model, train_loader, optimizer, criterion, device, epochs=10)
    evaluate(model, validate_loader, criterion, device)

    print("\nVisualizing original and reconstructed images from AutoEncoder...")
    models.visualize_reconstruction(autoencoder, validate_loader, device, num_images=6)
    # Example: sequence length = 50, features per step = 32
    rnn_model = models.RNNClassifier(input_size=128, hidden_size=64, num_layers=2, num_classes=5, bidirectional=True)
    print(rnn_model)
    mobilenet_model = models.MobileNetClassifier(num_classes=4)
    print(mobilenet_model)
    # Example: sequence length = 50, feature size = 32, num_classes = 5
    lstm_model = models.LSTMClassifier(
        input_size=32,
        hidden_size=64,
        num_layers=2,
        num_classes=5,
        bidirectional=True
    )
    print(lstm_model)


def load_and_visualize_dataset(path_to_data):
    """This is your original dataset loading code, but returns loaders and classes."""
    if not os.path.exists(path_to_data):
        raise FileNotFoundError(f"{path_to_data} does not exist!")

    train_loader, test_loader, validate_loader,classes = pre.load_data(
        data_dir=path_to_data,
        batch_size=32,
        augment=False
    )

    classes = train_loader.dataset.classes
    print(f"\nClasses: {classes}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(validate_loader.dataset)}")

    pre.visualize_data(train_loader, classes, num_samples=8)

    return train_loader, test_loader, validate_loader, classes



if __name__ == '__main__':
    main()
