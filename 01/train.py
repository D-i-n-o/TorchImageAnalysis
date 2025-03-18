import os
import torch
import torchvision.transforms as transforms
import csv

from torch.utils.data import DataLoader
from image_data import ImageData, CLASS_TO_IDX
from sklearn.metrics import average_precision_score, confusion_matrix
from Precode.ResNet import ResNet

NUM_CLASSES = 6
NUM_IMAGES = 17034

def train(data_loader: DataLoader, model: torch.nn.Module, optimizer, criterion, device):
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

def compute_metrics(actuals, predictions):
    ap_scores = []
    conf_mat = confusion_matrix(actuals, predictions)
    accuracy_scores = [conf_mat[i, i] / conf_mat[i].sum() for i in range(NUM_CLASSES)]
    
    for cls in range(NUM_CLASSES):
        y_true = [1 if actual == cls else 0 for actual in actuals]
        y_score = [1 if prediction == cls else 0 for prediction in predictions]
        ap_scores.append(average_precision_score(y_true, y_score))
        
    return ap_scores, accuracy_scores

def save_metrics(output_file, ap_scores, accurary_scores, epoch_num):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Class', 'Average Precision', 'Accuracy'])
        for cls in range(NUM_CLASSES):
            writer.writerow([epoch_num, cls, ap_scores[cls], accurary_scores[cls]])
            
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
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    image_transform = transforms.Compose([
        transforms.Resize((150, 150)),
    ])

    train_data = ImageData(
        root_dir = data_dir,
        data_dir = train_dir,
        transform = image_transform
    )

    model = ResNet(
        img_channels = 3, 
        num_layers = 18,
        num_classes = NUM_CLASSES
    )

    data_loader = DataLoader(train_data, batch_size = 32, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        running_loss, predictions, actuals = train(data_loader, model, optimizer, criterion, device)
        ap_scores, accurary_scores = compute_metrics(actuals, predictions)
        output_file = os.path.join(script_dir, 'metrics.csv')
        save_metrics(output_file, ap_scores, accurary_scores, epoch)
        torch.save(model.state_dict(), model_file)
        model_file = os.path.join(script_dir, f'model_epoch_{epoch}.pth')
        
        print(f'Epoch {epoch + 1}, Loss: {running_loss}')
        
    model.eval()

    val_data = ImageData(
        root_dir = data_dir,
        data_dir = val_dir,
        transform = image_transform
    )
    test_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    all_actuals, all_predictions = predict(model, test_loader, device)
    val_ap_scores, val_accuracy_scores = compute_metrics(all_actuals, all_predictions)
    output_file = os.path.join(script_dir, 'val_metrics.csv')
    save_metrics(output_file, val_ap_scores, val_accuracy_scores, 'val')