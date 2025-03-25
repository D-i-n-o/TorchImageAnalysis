import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import csv

from torch.utils.data import DataLoader
from image_data import ImageData, NUM_CLASSES, NUM_IMAGES
from Precode.ResNet import ResNet
from evaluate import compute_metrics, save_class_metrics, predict, evaluate_model

NUM_EPOCHS = 20


def get_optimizer(model, name='Adam'):
    optimizers = {
        'Adam': torch.optim.Adam(model.parameters(), lr=0.001),
        'SGD': torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
        'RMSprop': torch.optim.RMSprop(model.parameters(), lr=0.001)
    }
    return optimizers[name]

def train(data_loader: DataLoader, model: torch.nn.Module, optimizer, criterion, device):
    torch.manual_seed(42)
    model.train()
    running_loss = 0.0

    predictions = []
    actuals = []

    for i, (images, labels) in enumerate(data_loader): # iterates in batches
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        actuals.extend(labels.cpu().numpy())
        
        # correct += (predicted == labels).sum().item()
    
    return running_loss / len(data_loader), predictions, actuals
         
def predict(model, test_loader, device):
    all_predictions = []
    all_actuals = []
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_actuals.extend(labels.cpu().numpy())
    return all_actuals, all_predictions

if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    
    data_dir = os.path.join(script_dir, 'data')
    train_dir = os.path.join(data_dir, 'train')


    image_transform = transforms.Compose([
        transforms.Resize((150, 150)),
    ])

    train_data = ImageData(
        root_dir = data_dir,
        data_dir = train_dir,
        transform = image_transform
    )

    device = torch.device('cuda')
    model = models.resnet18(pretrained=True).to(device)
    model.fc = torch.nn.Linear(512, NUM_CLASSES).to(device)

    data_loader = DataLoader(train_data, batch_size = 32, shuffle = True)

    optimizer = get_optimizer(model, 'SGD')
    criterion = torch.nn.CrossEntropyLoss()
    
    running_losses = []
    model_dir = os.path.join(script_dir, 'model-R18-sgd-pretrained')
    os.makedirs(model_dir, exist_ok=True)
    for epoch in range(NUM_EPOCHS):
        running_loss, predictions, actuals = train(data_loader, model, optimizer, criterion, device)
        ap_scores, accurary_scores = compute_metrics(actuals, predictions)
        output_file = os.path.join(model_dir, 'train_metrics.csv')
        save_class_metrics(output_file, ap_scores, accurary_scores, epoch)
        if (epoch + 1) % 5 == 0:
            model_file = os.path.join(model_dir, f'epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), model_file)
        
        running_losses.append(running_loss)
        print(f'Epoch {epoch + 1}, Loss: {running_loss}')
    
    losses_file = os.path.join(model_dir, 'running_losses.txt')
    with open(losses_file, 'w') as f:
        for epoch, loss in enumerate(running_losses, start=1):
            f.write(f'Epoch {epoch}: Loss {loss}\n')
        


    



    