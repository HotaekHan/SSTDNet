import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.autograd import Variable

from sstdnet import load_sstdnet
from encoder import DataEncoder
from PIL import Image, ImageDraw

import os
import argparse
import multiprocessing as mp
import sys
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch SSTDNet Test')
parser.add_argument('--ckpt_path', '-ckpt', default='checkpoint', type=str, help='checkpoint path')
parser.add_argument('--image_path', '-image', default='../test', type=str, help='test image path')
parser.add_argument('--num_processes', '-np', default=10, type=int, help='num. of processor')
args = parser.parse_args()

def find_best_model(path):
    ckpt_file = dict()
    minimum_loss = float('inf')
    minimum_idx = 0
    for (path, dirs, files) in os.walk(path):
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

    return minimum_idx, minimum_loss



def read_images(path, num_processes):
    print('Loading image..')

    img_files = list()
    for (path, dir, files) in os.walk(path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]

            if ext == '.jpg':
                img_files.append(path + '/' + filename)

    num_of_imgs_each_bin = int(img_files.__len__() / num_processes)
    bins_of_images = list()

    for start_idx in range(0, num_processes * num_of_imgs_each_bin, num_of_imgs_each_bin):
        if start_idx == num_of_imgs_each_bin * (num_processes - 1):
            bins_of_images.append(img_files[start_idx:])
        else:
            bins_of_images.append(img_files[start_idx:start_idx + num_of_imgs_each_bin])

    return bins_of_images

def prediction(bin_of_images, checkpoint_dir, minimum_idx, result_dir):
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

    for img_file in bin_of_images:
        img = Image.open(img_file)
        w = img.width
        h = img.height

        print('Predicting : ' + img_file)
        x = transform(img)
        x = x.unsqueeze(0)
        x = Variable(x, volatile=True)
        loc_preds, cls_preds, mask_pred = net(x)

        # print('Decoding..')
        encoder = DataEncoder()
        boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))

        draw = ImageDraw.Draw(img)

        img_file_name = img_file.split("/")[-1]
        txt_file_name = img_file_name.replace(".jpg", ".result")

        result_txt = open(result_dir+"/"+txt_file_name, 'w')

        for result_idx in range(0, boxes.__len__(), 1):
            draw.rectangle(list(boxes[result_idx]), outline='red')
            result_txt.write(str(boxes[result_idx][0])+"\t"+str(boxes[result_idx][1])+"\t"+str(boxes[result_idx][2])+"\t"+str(boxes[result_idx][3])
                             +"\t"+str(labels[result_idx])+"\n")
        result_txt.close()

        img.save(result_dir+"/"+img_file_name)

        mask_pred = F.softmax(mask_pred)
        mask_data = mask_pred.data.numpy()
        mask_data = mask_data[:, 1:2, :, :]
        mask_data = np.squeeze(mask_data)
        mask_img = Image.fromarray(np.uint8(mask_data * 255.), 'L')
        mask_img.save(result_dir + "//" + img_file_name.replace(".jpg", ".png"))



if __name__ =='__main__':
    assert sys.version_info >= (3, 4, 0)

    result_dir = 'result_multi'
    # image_path =
    num_processes = args.num_processes

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    procs = list()
    best_model_idx, best_model_loss = find_best_model(args.ckpt_path)
    bins_of_images = read_images(args.image_path, num_processes)

    mp.set_start_method('spawn')
    for bin_of_img in bins_of_images:
        proc = mp.Process(target=prediction, args=(bin_of_img, args.ckpt_path, best_model_idx, result_dir,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
