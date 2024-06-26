import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.multiprocessing as mp
import torch.distributed as dist

import os
import time
import argparse

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout2d(0.10)
        self.dp2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4608, 64) # 4608 is basically 12 X 12 X 32
        self.fc2 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        op = F.log_softmax(x, dim=1)
        return op

def train(gpu_num, args):
    rank = args.machine_id * args.num_gpu_processes + gpu_num                        
    dist.init_process_group(                                   
    backend='nccl',                                         
    init_method='env://',                                   
    world_size=args.world_size,                              
    rank=rank                                               
    ) 
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu_num)
    model.cuda(gpu_num)
    criterion = nn.NLLLoss().cuda(gpu_num) # nll is the negative likelihood loss
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1302,), (0.3069,))]))  
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    train_dataloader = torch.utils.data.DataLoader(
       dataset=train_dataset,
       batch_size=args.batch_size,
       shuffle=False,            
       num_workers=0,
       pin_memory=True,
       sampler=train_sampler)
    optimizer = optim.Adadelta(model.parameters(), lr=0.5)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu_num])
    model.train()
    for epoch in range(args.epochs):
        for b_i, (X, y) in enumerate(train_dataloader):
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            pred_prob = model(X)
            loss = criterion(pred_prob, y) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if b_i % 10 == 0 and gpu_num==0:
                print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                    epoch, b_i, len(train_dataloader),
                    100. * b_i / len(train_dataloader), loss.item()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-machines', default=1, type=int,)
    parser.add_argument('--num-gpu-processes', default=1, type=int)
    parser.add_argument('--machine-id', default=0, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    args = parser.parse_args()
    
    args.world_size = args.num_gpu_processes * args.num_machines                
    os.environ['MASTER_ADDR'] = '127.0.0.1'              
    os.environ['MASTER_PORT'] = '8892'      
    start = time.time()
    mp.spawn(train, nprocs=args.num_gpu_processes, args=(args,))
    print(f"Finished training in {time.time()-start} secs")

if __name__ == '__main__':
    main()
