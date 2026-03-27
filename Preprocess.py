import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# Consistent normalization used across ALL splits and inference
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD  = [0.5, 0.5, 0.5]


def load_data(data_dir, batch_size=32, augment=False):
    norm = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1),
            transforms.ToTensor(),
            norm
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            norm
        ])

    # Validation and test always use clean transform (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        norm
    ])

    train_data    = datasets.ImageFolder(root=os.path.join(data_dir, 'train'),    transform=train_transform)
    test_data     = datasets.ImageFolder(root=os.path.join(data_dir, 'test'),     transform=eval_transform)
    validate_data = datasets.ImageFolder(root=os.path.join(data_dir, 'validate'), transform=eval_transform)

    print(f"Found {len(train_data.classes)} classes: {train_data.classes}")
    print(f"Train dataset size:      {len(train_data)}")
    print(f"Test dataset size:       {len(test_data)}")
    print(f"Validation dataset size: {len(validate_data)}")

    train_loader    = DataLoader(train_data,    batch_size=batch_size, shuffle=True)
    test_loader     = DataLoader(test_data,     batch_size=batch_size, shuffle=False)
    validate_loader = DataLoader(validate_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, validate_loader, train_data.classes


def visualize_data(train_loader, classes, num_samples=5):
    """Visualize sample images from the training dataset."""
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    num_samples = min(num_samples, len(images))
    images = images[:num_samples]
    labels = labels[:num_samples]

    images_np = images.numpy().transpose(0, 2, 3, 1)

    # Denormalize using the same mean/std used during loading
    mean = np.array(NORM_MEAN)
    std  = np.array(NORM_STD)
    images_np = images_np * std + mean
    images_np = np.clip(images_np, 0, 1)

    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        axes[i].imshow(images_np[i])
        axes[i].axis('off')
        axes[i].set_title(f"{classes[labels[i].item()]}", fontsize=10, pad=10)

    plt.tight_layout()
    plt.show()

    print(f"\nDisplayed {num_samples} sample images:")
    for i in range(num_samples):
        print(f"  Image {i + 1}: {classes[labels[i].item()]}")


def get_class_distribution(data_loader, classes):
    """Get the distribution of classes in the dataset."""
    class_counts = {class_name: 0 for class_name in classes}
    for _, labels in data_loader:
        for label in labels:
            class_name = classes[label.item()]
            class_counts[class_name] += 1
    return class_counts


def visualize_class_distribution(train_loader, test_loader, validate_loader, classes):
    """Visualize the distribution of classes across train, test, and validation sets."""
    train_dist = get_class_distribution(train_loader, classes)
    test_dist  = get_class_distribution(test_loader,  classes)
    val_dist   = get_class_distribution(validate_loader, classes)

    x     = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    train_counts = [train_dist[cls] for cls in classes]
    test_counts  = [test_dist[cls]  for cls in classes]
    val_counts   = [val_dist[cls]   for cls in classes]

    ax.bar(x - width, train_counts, width, label='Train',      alpha=0.8)
    ax.bar(x,         test_counts,  width, label='Test',       alpha=0.8)
    ax.bar(x + width, val_counts,   width, label='Validation', alpha=0.8)

    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Images')
    ax.set_title('Class Distribution Across Train/Test/Validation Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nClass Distribution:")
    print("-" * 50)
    print(f"{'Class':<15} {'Train':<8} {'Test':<8} {'Val':<8} {'Total':<8}")
    print("-" * 50)

    for cls in classes:
        train_count = train_dist[cls]
        test_count  = test_dist[cls]
        val_count   = val_dist[cls]
        total_count = train_count + test_count + val_count
        print(f"{cls:<15} {train_count:<8} {test_count:<8} {val_count:<8} {total_count:<8}")

    total_train = sum(train_dist.values())
    total_test  = sum(test_dist.values())
    total_val   = sum(val_dist.values())
    grand_total = total_train + total_test + total_val

    print("-" * 50)
    print(f"{'TOTAL':<15} {total_train:<8} {total_test:<8} {total_val:<8} {grand_total:<8}")