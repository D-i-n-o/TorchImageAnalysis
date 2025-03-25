import torch
import os
import csv

from sklearn.metrics import average_precision_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from Precode.ResNet import ResNet
from torchvision import models


from image_data import ImageData, NUM_CLASSES


def evaluate_model(model, data_loader, device, output_file, epoch_num):
    model.eval()
    all_actuals, all_predictions = predict(model, data_loader, device)
    ap_scores, accuracy_scores = compute_metrics(all_actuals, all_predictions)
    save_class_metrics(output_file, ap_scores, accuracy_scores, str(epoch_num))

def save_class_metrics(output_file, ap_scores, accurary_scores, epoch_num):
    file_exists = os.path.isfile(output_file)
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Epoch', 'Class', 'Average Precision', 'Accuracy'])
        for cls in range(NUM_CLASSES):
            writer.writerow([epoch_num, cls, ap_scores[cls], accurary_scores[cls]])    

def compute_metrics(actuals, predictions):
    ap_scores = []
    conf_mat = confusion_matrix(actuals, predictions)
    accuracy_scores = [conf_mat[i, i] / conf_mat[i].sum() for i in range(NUM_CLASSES)]
    
    for cls in range(NUM_CLASSES):
        y_true = [1 if actual == cls else 0 for actual in actuals]
        y_score = [1 if prediction == cls else 0 for prediction in predictions]
        ap_scores.append(average_precision_score(y_true, y_score))
    
    return ap_scores, accuracy_scores

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
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    image_transform = transforms.Compose([
        transforms.Resize((150, 150)),
    ])
    
    val_data = ImageData(
        root_dir = data_dir,
        data_dir = val_dir,
        transform = image_transform
    )
    data_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    test_data = ImageData(
        root_dir = data_dir,
        data_dir = test_dir,
        transform = image_transform
    )
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)    
    
    device = torch.device('cuda')
    model = models.resnet18(pretrained=True).to(device)
    model.fc = torch.nn.Linear(512, NUM_CLASSES).to(device)
    
    model_dir = os.path.join(script_dir, 'model-R18-sgd-pretrained')
    for epoch in [10, 15, 20]:
        model_file = os.path.join(model_dir, f'epoch_{epoch}.pt')
        model.load_state_dict(torch.load(model_file))
        evaluate_model(model, data_loader, device, os.path.join(model_dir, 'val_metrics.csv'), epoch)
        evaluate_model(model, test_loader, device, os.path.join(model_dir, 'test_metrics.csv'), epoch)