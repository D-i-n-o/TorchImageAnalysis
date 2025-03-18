import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

NUM_CLASSES = 6
NUM_IMAGES = 17034

CLASS_TO_IDX = {
    'buildings': 0,
    'forest': 1, 
    'glacier': 2,
    'mountain': 3,
    'sea': 4,
    'street': 5,
}

class ImageData(Dataset):
    
    def __init__(self, root_dir, data_dir, transform=None):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.absolut_data_dir = os.path.join(root_dir, data_dir)
        
        self.transform = transform
        self.images = []
        self.labels = []
        
        for cls in os.listdir(self.absolut_data_dir):
            cls_dir = os.path.join(self.absolut_data_dir, cls)
            for img in os.listdir(cls_dir):
                self.images.append(img)
                self.labels.append(cls)

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.absolut_data_dir, self.labels[idx], self.images[idx]))

        if self.transform:
            image = self.transform(image)
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        return image, CLASS_TO_IDX[self.labels[idx]]