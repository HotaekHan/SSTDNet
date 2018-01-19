import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.autograd import Variable

from sstdnet import load_sstdnet
from encoder import DataEncoder
from PIL import Image, ImageDraw

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch SSTDNet Test')
parser.add_argument('--path', '-p', default='checkpoint', type=str, help='checkpoint path')
args = parser.parse_args()

ckpt_file = dict()
minimum_loss = float('inf')
minimum_idx = 0
checkpoint_dir = args.path
for (path, dirs, files) in os.walk(checkpoint_dir):
    for filename in files:
        ext = os.path.splitext(filename)[-1]

        if ext == '.pth':
            if torch.cuda.is_available():
                load_pth = torch.load(path+"//"+filename)
            else:
                load_pth = torch.load(path + "//" + filename, map_location=lambda storage, loc: storage)
            valid_loss = load_pth['loss']

            ckpt_idx = filename
            ckpt_idx = int(ckpt_idx.split("-")[-1].split(".")[0])

            ckpt_file[ckpt_idx] = valid_loss

            if valid_loss < minimum_loss:
                minimum_idx = ckpt_idx
                minimum_loss = valid_loss

for idx in ckpt_file:
    print("ckpt-"+str(idx)+" "+str(ckpt_file[idx]))

print('Loading model..')

if torch.cuda.is_available():
    load_pth = torch.load(checkpoint_dir + "/ckpt-" + str(minimum_idx) + ".pth")
else:
    load_pth = torch.load(checkpoint_dir + "/ckpt-" + str(minimum_idx) + ".pth",
                          map_location=lambda storage, loc: storage)

valid_loss = load_pth['loss']
print("valid loss : " + str(valid_loss))

num_classes = load_pth['num_classes']
num_batch = load_pth['batch']
num_crops = load_pth['crops']
print("num. batch : " + str(num_batch))
print("num. crops : " + str(num_crops))

net = load_sstdnet(num_classes=num_classes, using_pretrained=False)
net.load_state_dict(load_pth['net'])
net.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

result_dir = 'result'

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

print('Loading image..')

img_files = list()
for (path, dir, files) in os.walk('../test'):
    for filename in files:
        ext = os.path.splitext(filename)[-1]

        if ext == '.jpg':
            img_files.append(path+'//'+filename)

import datetime
avg_elap_pred = 0.0
avg_elap_decode = 0.0

encoder = DataEncoder()
for img_file in img_files:
    img = Image.open(img_file)
    w = img.width
    h = img.height

    print('Predicting..')
    start_pred = datetime.datetime.now()
    x = transform(img)
    x = x.unsqueeze(0)
    x = Variable(x, volatile=True)
    loc_preds, cls_preds, mask_pred = net(x)
    end_pred = datetime.datetime.now()
    elapsed_pred = end_pred - start_pred
    ms_elapsed_pred = elapsed_pred.total_seconds() * 1000
    avg_elap_pred += ms_elapsed_pred
    print("Elapsed time of pred : " + str(ms_elapsed_pred) + "ms")


    print('Decoding..')
    start_decode = datetime.datetime.now()
    boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))
    end_decode = datetime.datetime.now()
    elapsed_decode = end_decode - start_decode
    ms_elapsed_decode = elapsed_decode.total_seconds() * 1000
    avg_elap_decode += ms_elapsed_decode
    print("Elapsed time of decode : " + str(ms_elapsed_decode) + "ms")

    draw = ImageDraw.Draw(img)

    img_file_name = img_file.split("//")[-1]
    txt_file_name = img_file_name.replace(".jpg", ".result")

    result_txt = open(result_dir+"//"+txt_file_name, 'w')

    for box in boxes:
        draw.rectangle(list(box), outline='red')
        result_txt.write(str(box[0])+"\t"+str(box[1])+"\t"+str(box[2])+"\t"+str(box[3])+"\n")
    result_txt.close()

    img.save(result_dir+"//"+img_file_name)

    mask_pred = F.softmax(mask_pred)
    mask_data = mask_pred.data.numpy()
    mask_data = mask_data[:, 1:2, :, :]
    mask_data = np.squeeze(mask_data)
    mask_img = Image.fromarray(np.uint8(mask_data * 255.), 'L')
    mask_img.save(result_dir + "//" + img_file_name.replace(".jpg", ".png"))

num_of_image = float(img_files.__len__())

avg_elap_pred = avg_elap_pred / num_of_image
avg_elap_decode = avg_elap_decode / num_of_image

print("Avg. elapsed time of pred : " + str(avg_elap_pred) + "ms")
print("Avg. elapsed time of decode : " + str(avg_elap_decode) + "ms")
