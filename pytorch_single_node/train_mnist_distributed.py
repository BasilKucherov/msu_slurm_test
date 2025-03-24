import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def setup(rank, world_size, backend='nccl'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = os.environ.get('SLURM_LAUNCH_NODE_IPADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup():
    """Clean up distributed environment."""
    dist.destroy_process_group()


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


def train(model, device, train_loader, optimizer, epoch, rank, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if rank == 0 and batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data) * dist.get_world_size()}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader, rank):
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

    if dist.is_initialized():
        test_loss_tensor = torch.tensor([test_loss]).to(device)
        correct_tensor = torch.tensor([correct]).to(device)

        dist.reduce(test_loss_tensor, 0, op=dist.ReduceOp.SUM)
        dist.reduce(correct_tensor, 0, op=dist.ReduceOp.SUM)

        test_loss = test_loss_tensor.item()
        correct = correct_tensor.item()

    if rank == 0:
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
              f' ({accuracy:.1f}%)\n')
        return accuracy
    return 0


def save_checkpoint(state, filename, rank):
    if rank == 0:
        print(f"=> Saving checkpoint to {filename}")
        torch.save(state, filename)


def load_checkpoint(filename, model, optimizer, rank):
    if not os.path.isfile(filename):
        if rank == 0:
            print(f"Checkpoint {filename} not found, starting from scratch")
        return 0, 0

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(filename, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if rank == 0:
        print(f"=> Loading checkpoint from {filename}")
        print(f"Resuming from epoch {checkpoint['epoch']} with best accuracy: {checkpoint['best_accuracy']:.2f}%")

    return checkpoint['epoch'], checkpoint['best_accuracy']


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example with Distributed Training')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, help='Learning rate step gamma')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_distributed',
                        help='directory to save checkpoints')
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoint file')
    parser.add_argument('--data-dir', type=str, default='/workspace/data', help='path to store MNIST dataset')
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')
    args = parser.parse_args()

    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    global_rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        print("CUDA is not available, this will likely not work well with DDP!")
        return

    print(f"Initializing process group: rank={global_rank}, world_size={world_size}")
    setup(global_rank, world_size, args.backend)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed)

    if global_rank == 0:
        print(f"World size: {world_size}, Rank: {global_rank}, Local rank: {local_rank}")
        print(f"Using {torch.cuda.device_count()} GPUs per node")
        print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    os.makedirs(args.data_dir, exist_ok=True)
    if global_rank == 0:
        print(f"Using dataset path: {args.data_dir}")

    train_dataset = datasets.MNIST(args.data_dir, train=True, download=(global_rank == 0), transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = Net().to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_accuracy = 0
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.resume)
        start_epoch, best_accuracy = load_checkpoint(checkpoint_path, model, optimizer, global_rank)

    total_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()

        train(model, device, train_loader, optimizer, epoch + 1, global_rank)
        accuracy = test(model, device, test_loader, global_rank)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if global_rank == 0:
            print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds (DistributedDataParallel)")

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': max(best_accuracy, accuracy)
            }
            save_checkpoint(checkpoint, os.path.join(args.checkpoint_dir,
                            f'checkpoint_epoch_{epoch+1}.pt'), global_rank)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_checkpoint(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pt'), global_rank)

        scheduler.step()

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    if global_rank == 0:
        print(f"\nTotal training time: {total_duration:.2f} seconds")
        print(f"Average time per epoch: {total_duration / (args.epochs - start_epoch):.2f} seconds")
        print(
            f"Training completed using {torch.cuda.device_count()} GPUs per node across {world_size//torch.cuda.device_count()} nodes")

    cleanup()


if __name__ == '__main__':
    main()
