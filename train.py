from __future__ import print_function

import os
import argparse

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision.transforms as transforms

from loss import FocalLoss
from sstdnet import load_sstdnet
from datagen import ListDataset

from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch SSTDnet Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--resume_path', type=str, default="checkpoint/ckpt-0.pth", help='checkpoint path')
parser.add_argument('--num_classes', '-classes', type=int, help='num. of classes')
parser.add_argument('--optimizer', '-op', type=str, default='SGD', help='SGD or Adam')
parser.add_argument('--num_crops', '-nc', type=int, default=2, help='How many crops')
parser.add_argument('--batch_size', '-batch', type=int, default=8, help='Batch size')
args = parser.parse_args()

lr = args.lr
num_classes = args.num_classes
selected_optim = args.optimizer
n_crops = args.num_crops
batch_size = args.batch_size
using_gpu = torch.cuda.is_available()

checkpoint_dir = 'checkpoint'

start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = ListDataset(root="../train", gt_extension=".txt",
                      labelmap_path="class_label_map.xlsx", is_train=True, transform=transform, input_image_size=512,
                      num_crops=n_crops, original_img_size=512)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

validset = ListDataset(root="../valid", gt_extension=".txt",
                      labelmap_path="class_label_map.xlsx", is_train=False, transform=transform, input_image_size=512,
                      num_crops=5, original_img_size=512)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=validset.collate_fn)

print("lr : " + str(lr))
print("num. of classes : " + str(num_classes))
print("optimizer : " + selected_optim)
print("Using cuda : " + str(using_gpu))
print("Num. of crops : " + str(n_crops))
print("Size of batch : " + str(batch_size))
print("num. train data : " + str(trainset.__len__()))
raw_input("Press any key to continue..")

parameters_out = open("params.txt", 'w')
parameters_out.write("lr: "+str(lr)+"\n")
parameters_out.write("num. classes: "+str(num_classes)+"\n")
parameters_out.write("optimizer: "+selected_optim+"\n")
parameters_out.write("num. crops: "+str(n_crops)+"\n")
parameters_out.write("batch size: "+str(batch_size)+"\n")
parameters_out.write("num. training sample: "+str(trainset.__len__())+"\n")
parameters_out.close()

# Model
net = load_sstdnet(num_classes=num_classes, using_pretrained=True)
num_parameters = 0.
for param in net.parameters():
    sizes = param.size()

    num_layer_param = 1.
    for size in sizes:
        num_layer_param *= size
    num_parameters += num_layer_param

print(net)
print("num. of parameters : " + str(num_parameters))

if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume_path)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

criterion = FocalLoss(num_classes=net.num_classes)

if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)
if using_gpu is True:
    net.cuda()

if selected_optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
elif selected_optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=lr)
else:
    print("not supported optimizer")

# step_exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    avg_matched_anchor = 0.
    # step_exp_lr_scheduler.step()
    for batch_idx, (inputs, loc_targets, cls_targets, mask_targets) in enumerate(trainloader):

        if using_gpu is True:
            inputs = Variable(inputs.cuda())
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())
            mask_targets = Variable(mask_targets.cuda())
        else:
            inputs = Variable(inputs)
            loc_targets = Variable(loc_targets)
            cls_targets = Variable(cls_targets)
            mask_targets = Variable(mask_targets)

        optimizer.zero_grad()
        loc_preds, cls_preds, mask_preds = net(inputs)
        loc_loss, cls_loss, mask_loss, num_matched_anchors = \
            criterion(loc_preds, loc_targets, cls_preds, cls_targets, mask_preds, mask_targets)
        loss = ((loc_loss + cls_loss) / num_matched_anchors) + mask_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        avg_matched_anchor += float(num_matched_anchors)
        print('epoch: %3d | iter: %4d | loc_loss: %.3f | cls_loss: %.3f | mask_loss: %.3f | train_loss: %.3f | avg_loss: %.3f | avg_num. matched: %d'
              % (epoch, batch_idx, loc_loss.data[0] / num_matched_anchors, cls_loss.data[0] / num_matched_anchors,
                 mask_loss.data[0], loss.data[0], train_loss / (batch_idx + 1), avg_matched_anchor / (batch_idx + 1)))

# Test
def valid(epoch):
    print('\nValid')
    net.eval()
    valid_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets, mask_targets) in enumerate(validloader):

        if using_gpu is True:
            inputs = Variable(inputs.cuda(), volatile=True)
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())
            mask_targets = Variable(mask_targets.cuda())
        else:
            inputs = Variable(inputs, volatile=True)
            loc_targets = Variable(loc_targets)
            cls_targets = Variable(cls_targets)
            mask_targets = Variable(mask_targets)

        loc_preds, cls_preds, mask_preds = net(inputs)
        loc_loss, cls_loss, mask_loss, num_matched_anchors = \
            criterion(loc_preds, loc_targets, cls_preds, cls_targets, mask_preds, mask_targets)
        loss = ((loc_loss + cls_loss) / num_matched_anchors) + mask_loss
        valid_loss += loss.data[0]
        print('loc_loss: %.3f | cls_loss: %.3f | valid_loss: %.3f | avg_loss: %.3f'
              % (loc_loss.data[0] / num_matched_anchors, cls_loss.data[0] / num_matched_anchors,
                 loss.data[0], valid_loss / (batch_idx + 1)))

    # Save checkpoint
    # Every checkpoints are stored to analyze how is going training
    # Model is selected by low-validation error.
    valid_loss /= len(validloader)
    print('Saving..')
    state = {
        'net': net.module.state_dict(),
        'loss': valid_loss,
        'epoch': epoch,
        'num_classes': num_classes,
        'lr': lr,
        'batch': batch_size,
        'crops': n_crops,
        'op': selected_optim

    }
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state, checkpoint_dir+'/ckpt-'+str(epoch)+'.pth')


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    valid(epoch)
