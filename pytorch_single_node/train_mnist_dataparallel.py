import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.1f}%)\n')
    return accuracy


def save_checkpoint(state, filename):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint from {filename}")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['best_accuracy']
    return 0, 0


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example with DataParallel')
    parser.add_argument('--batch-size', type=int, default=256, help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, help='Learning rate step gamma')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoint file')
    parser.add_argument('--data-dir', type=str, default='/workspace/data', help='path to store MNIST dataset')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} CUDA device(s)")
        for i in range(num_gpus):
            print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Using CPU")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    os.makedirs(args.data_dir, exist_ok=True)
    print(f"Using dataset path: {args.data_dir}")

    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=4)

    model = Net().to(device)

    if use_cuda and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_accuracy = 0
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.resume)
        if os.path.exists(checkpoint_path):
            start_epoch, best_accuracy = load_checkpoint(checkpoint_path, model, optimizer)
            print(f"Resuming from epoch {start_epoch} with best accuracy: {best_accuracy:.2f}%")
        else:
            print(f"Checkpoint {args.resume} not found, starting from scratch")

    total_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        train(model, device, train_loader, optimizer, epoch + 1)
        accuracy = test(model, device, test_loader)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds (DataParallel)")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': max(best_accuracy, accuracy)
        }
        save_checkpoint(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pt'))

        scheduler.step()

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nTotal training time: {total_duration:.2f} seconds")
    print(f"Average time per epoch: {total_duration / (args.epochs - start_epoch):.2f} seconds")

    if torch.cuda.is_available():
        print(f"Training completed using {torch.cuda.device_count()} GPUs with DataParallel")


if __name__ == '__main__':
    main()
