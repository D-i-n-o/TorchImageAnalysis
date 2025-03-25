import csv
import os
import matplotlib.pyplot as plt


def mean_average_precision(csv_file, epoch=None):
    ap_scores = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if epoch is not None and int(row['Epoch']) != epoch:
                continue
            ap_scores.append(float(row['Average Precision']))
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0

def mean_accuracy(csv_file, epoch=None):
    accuracy_scores = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if epoch is not None and int(row['Epoch']) != epoch:
                continue
            accuracy_scores.append(float(row['Accuracy']))
    return sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0

def plot_class_precisions(csv_file, save_to, epoch = None):
    plt.xlabel('Class')
    plt.ylabel('Average Precision')
    plt.ylim(0, 1.1)
    classes = []
    values = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if epoch is not None and int(row['Epoch']) != epoch:
                continue
            print(row)
            classes.append(row['Class'])
            values.append(float(row['Average Precision']))
    plt.bar(classes, values, color='skyblue')
    suffix = f'after {epoch} epochs' if epoch is not None else ''
    model_name = os.path.dirname(csv_file).split('/')[-1].replace('model-', '')
    title = f'Average Precisions by Class for  {model_name} on test set {suffix}'
    plt.title(title)
    # mean_ap = sum(values) / len(values) if values else 0
    # plt.text(
    #     0.5, 1.05,
    #     f'Mean Average Precision: {mean_ap:.4f}',
    #     ha='center', va='center', transform=plt.gca().transAxes, fontsize=10, color='black'
    # )
    plt.savefig(save_to)
    plt.close()
    
def plot_class_accuracies(csv_file, save_to, epoch = None):
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    classes = []
    values = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if epoch is not None and int(row['Epoch']) != epoch:
                continue
            classes.append(row['Class'])
            values.append(float(row['Accuracy']))
    plt.bar(classes, values, color='lightgreen')
    suffix = f'after {epoch} epochs' if epoch is not None else ''
    model_name = os.path.dirname(csv_file).split('/')[-1].replace('model-', '')
    title = f'Accuracies by Class for {model_name} on test set {suffix}'
    plt.title(title)
    # mean_acc = sum(values) / len(values) if values else 0
    # plt.text(
    #     0.5, 1.05,
    #     f'Mean Accuracy: {mean_acc:.4f}',
    #     ha='center', va='center', transform=plt.gca().transAxes, fontsize=10, color='black'
    # )
    plt.savefig(save_to)
    plt.close()
    
    
def plot_losses(csv_files, save_to):
    for csv_file in csv_files:
        epochs = []
        losses = []
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                epochs.append(int(row['Epoch']))
                losses.append(float(row['Loss']))
        label = os.path.dirname(csv_file).split('/')[-1].replace('model-', '')
        plt.plot(epochs, losses, marker='o', label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_to)
    plt.close()
    
    



if __name__ == '__main__':
    models = ['model-R18-adam', 'model-R18-sgd','model-R50-sgd']
    models = ['model-R18-sgd-pretrained']

    data_set = 'val'
    for model in models:
        csv_file = os.path.join(os.path.dirname(__file__), model, f'{data_set}_metrics.csv')
        for epoch in [10, 15, 20]:
            save_to = os.path.join(os.path.dirname(csv_file), f'{data_set}_average_precisions_{epoch}.png')
            plot_class_precisions(csv_file, save_to, epoch)
            save_to = os.path.join(os.path.dirname(csv_file), f'{data_set}_accuracies_{epoch}.png')
            plot_class_accuracies(csv_file, save_to, epoch)
            print(f'Model: {model}, Epoch: {epoch}')
            print(f'Mean Average Precision: {mean_average_precision(csv_file, epoch)}')
            print(f'Mean Accuracy: {mean_accuracy(csv_file, epoch)}')

    loss_files = [os.path.join(os.path.dirname(__file__), model, 'running_losses.csv') for model in models]
    save_to = os.path.join(os.path.dirname(loss_files[0]), 'pretrained_loss.png')
    plot_losses(loss_files, save_to)

