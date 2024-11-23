import torch
from torch import nn, optim
from torch.utils.data import Subset
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import random
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
import os


# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def train_model(model, optimizer, criterion, dataloader, num_epochs, train_scheduler, warmup_scheduler, warmup_epochs=20): 

    model.train()

    metrics_df = pd.DataFrame(columns=['Epoch', 'TrainingLoss', 'LearningRate', 'TrainingAccuracy'])

    for epoch in range(num_epochs):
        if epoch+1 > warmup_epochs:
            train_scheduler.step()

        running_loss = 0
        correct_predictions = 0
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
            running_loss += loss.item() * data.size(0)

            if epoch <= warmup_epochs:
                warmup_scheduler.step()

        epoch_loss = running_loss/len(dataloader.dataset)
        epoch_accuracy = correct_predictions/len(dataloader.dataset)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f'ResNet Model - Epoch {epoch+1}, Training Loss: {running_loss/len(dataloader.dataset):.5f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.5f}, Training Accuracy: {correct_predictions/len(dataloader.dataset):.5f}')
        
        metrics_df.loc[len(metrics_df)] = {
            'Epoch': epoch + 1,
            'TrainingLoss': round(epoch_loss,5),
            'LearningRate': round(current_lr,5),
            'TrainingAccuracy': epoch_accuracy
        }

    if not os.path.exists('result'):
        os.makedirs('result')
    metrics_df.to_csv(f'result/{args.model}_{args.batch_size}_{args.learning_rate}_{args.num_epoch}.csv', index=False)

@torch.no_grad()
def evaluate_model(net, loss_function, cifar100_test_loader):
    net.eval()
    test_loss = 0.0  
    correct = 0.0
    correct_top1 = 0
    correct_top5 = 0

    for images, labels in cifar100_test_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        top1, top5 = accuracy(outputs, labels, (1, 5))
        correct_top1 += top1
        correct_top5 += top5

    print("Evaluating Network.....")
    print(
        "Test set: Average loss: {:.4f}, Accuracy Top-1: {:.4f}, Accuracy Top-5: {:.4f}".format(
            test_loss / len(cifar100_test_loader.dataset),
            correct_top1 / len(cifar100_test_loader.dataset),
            correct_top5 / len(cifar100_test_loader.dataset),
        )
    )

    return correct_top1 / len(cifar100_test_loader.dataset), correct_top5 / len(cifar100_test_loader.dataset)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res


def test_class(model, dataloader, class_to_test):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            indices = (target == class_to_test)
            data, target = data[indices], target[indices]
            if len(data) == 0:
                continue
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
    if total == 0:
        print(f"No samples of class {class_to_test} in the test set.")
        return
    print(f'Test set: Class {class_to_test} Accuracy: {correct}/{total} ({100. * correct / total:.0f}%)') 
    

def test_all_classes(model, dataloader, num_classes):
    for class_to_test in range(num_classes):
        test_class(model, dataloader, class_to_test)


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    return cifar100_training

def get_test_dataloader(mean, std, batch_size, num_workers=2, shuffle=True):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    return cifar100_test


def get_dataloader(dataset, train, batch_size, num_workers=4):
    if dataset == 'CIFAR10':
        dataset = datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))
            ])
        )
    elif dataset == 'CIFAR100':
        if train:
            dataset = get_training_dataloader(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD, batch_size=batch_size)
        else:
            dataset = get_test_dataloader(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD, batch_size=batch_size)


    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


def main(args):
    # your code
    dataset = 'CIFAR100'
    load_model = args.load_model

    chosen_model = args.model
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epoch = args.num_epoch
    
    num_workers = 4
    input_size = 3 * 224 * 224

    if dataset == 'CIFAR10':
        output_size = 10
    elif dataset == 'CIFAR100':
        output_size = 100

    # Create dataloader - your code
    train_loader = get_dataloader(dataset, True, batch_size, num_workers)
    test_loader = get_dataloader(dataset, False, batch_size, num_workers)

    # total_model = resnet18(num_classes=output_size) - your code
    if chosen_model == 'resnet18':
        model = resnet18(num_classes=output_size)
    elif chosen_model == 'resnet50':
        model_50 = resnet50(num_classes=output_size)
    model.to(device)

    MILESTONES = [60, 120, 160]

    model_optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=MILESTONES, gamma=0.2)
    criterion = nn.CrossEntropyLoss()
    warmup_scheduler = WarmUpLR(model_optimizer, len(train_loader))
    warmup_epochs = 1
    

    # MILESTONES = [60, 120, 160]

    # your code - optimizer ...
    

    # Check if saved model parameters exist, if so load them
    if os.path.exists('best_model_params.pth') and (load_model == 1):
        model.load_state_dict(torch.load('best_model_params.pth'))
        print("Loaded saved model parameters.")
    else:
        best_accuracy = 0.0  # Initialize best accuracy
        start = datetime.now()
        train_model(model, model_optimizer, criterion, train_loader, num_epoch, train_scheduler, warmup_scheduler, warmup_epochs)

        # Evaluate total model
        total_accuracy, top5 = evaluate_model(model, criterion, test_loader)
        end = datetime.now()

        print(f'Total model on test set - Top1 Accuracy: {total_accuracy}, Top5 Accuracy: {top5}')
        print(f'learning time: {end - start}')

        # Save model results
        result_file = 'result/results.csv'
        if os.path.exists(result_file):
            results_df = pd.read_csv(result_file)
        else:
            results_df = pd.DataFrame(columns=['Model', 'BatchSize', 'LearningRate', 'Epochs', 'Top1Accuracy', 'Top5Accuracy', 'LearningTime'])
        
        results_df.loc[len(results_df)] = {
            'Model': chosen_model,
            'BatchSize': batch_size,
            'LearningRate': learning_rate,
            'Epochs': num_epoch,
            'Top1Accuracy': total_accuracy,
            'Top5Accuracy': top5,
            'LearningTime': end - start
        }
        results_df.to_csv(result_file, index=False)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ResNet-CIFAR100')
    # your code
    # parser.add_argument('--dataset', type=str, default='CIFAR100', help="This is dataset; MNIST, CIFAR10, CIFAR100")
    parser.add_argument('--batch_size', type=int, default=64, help="")
    parser.add_argument('--learning_rate', type=float, default='0.1', help="")
    parser.add_argument('--num_epoch', type=int, default=10, help="")
    parser.add_argument('--load_model', type=int, default=0, help="")
    parser.add_argument('--model', type=str, default='resnet18', help="")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # batch_size = args.batch_size
    # dataset = args.dataset
    # print(args.batch_size, args.dataset, args.learning_rate)
    main(args)