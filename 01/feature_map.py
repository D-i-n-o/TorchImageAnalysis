import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from image_data import ImageData

import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

    
def get_feature_map(image_path):
    def hook_fn(module, input, output):
        layer_name = list(selected_layers.keys())[list(selected_layers.values()).index(module)]
        feature_maps[layer_name] = output

    model = models.resnet18(pretrained=True)
    
    selected_layers = {
        'early': model.layer1[0].conv1,
        'middle': model.layer2[0].conv1,
        'late': model.layer4[0].conv1
    }
    feature_maps = {}

    for layer in selected_layers.values():
        layer.register_forward_hook(hook_fn)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_path = ''
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        model(image)
        
    return feature_maps
    
def visualize_feature_maps(feature_maps, image_index):
    for layer_name, feature_map in feature_maps.items():
        num_channels = feature_map.shape[1]
        fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 2, 2))
        for i in range(num_channels):
            ax = axes[i] if num_channels > 1 else axes
            ax.imshow(feature_map[0, i].cpu().numpy(), cmap='viridis')
            ax.axis('off')
        plt.savefig(f'feature_maps_image_{image_index}_{layer_name}.png')
        plt.close(fig)

if __name__ == '__main__':
    train_dir = ''
    data_loader = DataLoader(train_dir, batch_size = 10, shuffle = True)




    # Assuming train_dir is a directory containing images
    train_dir = '/path/to/your/train/directory'
    dataset = ImageData(train_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Retrieve a batch of images
    images, _ = next(iter(data_loader))

    # Process and visualize feature maps for the first 10 images
    for idx in range(10):
        image = images[idx].unsqueeze(0)
        feature_maps = get_feature_map(image)
        visualize_feature_maps(feature_maps, idx)
    
        
    # for layer_name, feature_map in feature_maps.items():
    #     print(f"Feature map from {layer_name}: {feature_map.shape}")
