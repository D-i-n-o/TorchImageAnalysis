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


def avg_non_positive_frac(dataloader, num_images=200):
    model = models.resnet18(pretrained=True)

    selected_layers = {
        'early': model.layer1[0].conv1,
        'middle': model.layer2[0].conv1,
        'late': model.layer4[0].conv1
    }

    feature_stats = {layer: [] for layer in selected_layers}

    def hook_fn(module, input, output):
        non_positive_percentage = (output <= 0).sum().item() / output.numel() * 100
        layer_name = list(selected_layers.keys())[list(selected_layers.values()).index(module)]
        feature_stats[layer_name].append(non_positive_percentage)


    for layer in selected_layers.values():
        layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i * images.size(0) >= num_images:
                break
            model(images)

    avg_stats = {layer: sum(stats) / (len(stats) if stats else 1) for layer, stats in feature_stats.items()}
    return avg_stats

    
def get_feature_map(image_tensor):
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

    model.eval()
    with torch.inference_mode():
        model(image_tensor)
        
    return feature_maps
    
def visualize_feature_maps(feature_maps, image_index, save_dir, max_channels=16):
    for layer_name, feature_map in feature_maps.items():
        num_channels = feature_map.shape[1]
        channels_to_plot = min(num_channels, max_channels)  # Limit the number of channels to visualize
        fig, axes = plt.subplots(1, channels_to_plot, figsize=(channels_to_plot * 2, 2))
        
        for i in range(channels_to_plot):
            ax = axes[i] if channels_to_plot > 1 else axes
            ax.imshow(feature_map[0, i].cpu().numpy(), cmap='viridis')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{layer_name}_image_{image_index}.png'))
        plt.close(fig)

if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    train_dir = os.path.join(script_dir, 'data', 'train')

    image_transform = transforms.Compose([
        transforms.Resize((150, 150)),
    ])

    train_data = ImageData(
        root_dir = train_dir,
        data_dir = train_dir,
        transform = image_transform
    )
    data_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    images, _ = next(iter(data_loader))
    
    
    maps_dir = os.path.join(script_dir, 'feature_maps')
    
    model = models.resnet18(pretrained=True)
    avg_stats = avg_non_positive_frac(data_loader, num_images=200)
    print(avg_stats)