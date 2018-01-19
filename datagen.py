from __future__ import print_function

import os
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
from encoder import DataEncoder

import xlrd
import numpy as np

class ListDataset(data.Dataset):
    def __init__(self, root, labelmap_path, gt_extension, is_train, transform, input_image_size, num_crops, original_img_size):
        '''
        Args:
          root: (str) ditectory to images.
          labelmap_path: (str) excel file path to contain classes
          gt_extension: (str) extension of gt file
          is_train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_image_size: (int) image size for train.
          num_crops: (int) how many crops the image(-1~5)
          original_img_size: (int) original image size
        '''
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.input_size = input_image_size
        self.original_img_size = original_img_size

        self.fnames = []
        self.offsets = []
        self.boxes = []
        self.labels = []

        self.data_encoder = DataEncoder()

        self.num_crops = num_crops

        # read label map
        workbook = xlrd.open_workbook(labelmap_path)
        work_sheet = workbook.sheet_by_index(0)

        num_of_rows = work_sheet.nrows

        label_map = []
        for idx in range(0, num_of_rows):
            label_map.append([work_sheet.row_values(idx)[0].encode('ascii', 'ignore'), work_sheet.row_values(idx)[1]])

        # read gt path
        gt_files = []
        img_files = []
        num_images = 0

        for (path, dir, files) in os.walk(self.root):
            for filename in files:
                ext = os.path.splitext(filename)[-1]

                if ext == gt_extension:
                    gt_files.append(filename)
                    img_file = filename.replace(ext, ".jpg")
                    img_files.append(self.root+"/"+img_file)
                    num_images += 1

        all_boxes = []
        all_labels = []

        # read gt files
        for gt_file in gt_files:
            f_read = open(self.root+"/"+gt_file, 'r')
            lines = f_read.readlines()

            box = []
            label = []

            for line in lines:
                split_line = line.split("\t")

                xmin = split_line[0]
                ymin = split_line[1]
                xmax = split_line[2]
                ymax = split_line[3]
                class_name = str(split_line[4].rstrip())
                class_idx = self.convert_from_name_to_label(class_name, label_map)

                if class_idx != 0:
                    box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                    label.append(int(class_idx))

            all_boxes.append(box)
            all_labels.append(label)

        if num_crops <= 0:
            for idx in range(0, num_images, 1):
                self.fnames.append(img_files[idx])
                self.boxes.append(torch.FloatTensor(all_boxes[idx]))
                self.labels.append(torch.LongTensor(all_labels[idx]))
        else:
            for idx in range(0, num_images, 1):
                ori_boxes = all_boxes[idx]
                ori_labels = all_labels[idx]
                offsets, crop_boxes, crop_labels = self.do_crop(ori_img_size=self.original_img_size,
                                                                target_img_size=self.input_size,
                                                                boxes=ori_boxes, labels=ori_labels)

                num_offsets = offsets.__len__()

                for idx_offset in range(0, num_offsets, 1):
                    self.fnames.append(img_files[idx])
                    self.offsets.append(offsets[idx_offset])
                    self.boxes.append(torch.FloatTensor(crop_boxes[idx_offset]))
                    self.labels.append(torch.LongTensor(crop_labels[idx_offset]))

        self.num_samples = self.fnames.__len__()

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        img = Image.open(fname)

        if self.num_crops <= 0:
            img, boxes = self.resize(img, boxes)
        else:
            offset = self.offsets[idx]
            crop_rect = (offset[0], offset[1], (offset[0]+self.input_size), (offset[1]+self.input_size))

            if offset[0] < 0 or offset[1] < 0:
                print("negative offset!")

            for box in boxes:
                if box[0] < 0 or box[1] < 0 or box[2] > self.input_size or box[3] > self.input_size:
                    print("negative box coordinate!")

            cropped_img = img.crop(crop_rect)
            img = cropped_img

        mask = torch.zeros(img.height, img.width).type(torch.LongTensor)

        for box in boxes:
            mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1

        # # Data augmentation while training.
        # if self.is_train:
        #     cropped_img, boxes = self.random_flip(cropped_img, boxes)
        #     cropped_img, boxes = self.scale_jitter(cropped_img, boxes)

        img = self.transform(img)
        return img, boxes, labels, mask

    def resize(self, img, boxes):
        '''Resize the image shorter side to input_size.

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) object boxes, sized [#obj, 4].

        Returns:
          (PIL.Image) resized image.
          (tensor) resized object boxes.

        Reference:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/utils/blob.py
        '''
        w = h = self.input_size
        ws = 1.0 * w / img.width
        hs = 1.0 * h / img.height
        scale = torch.Tensor([ws,hs,ws,hs])
        return img.resize((w,h)), scale*boxes

    def do_crop(self, ori_img_size, target_img_size, boxes, labels):
        num_boxes = boxes.__len__()
        num_labels = labels.__len__()

        if num_boxes != num_labels:
            print("error occur: Random crop")

        rand_indices = [0, 1, 2, 3, 4]
        np.random.shuffle(rand_indices)

        output_offsets = []
        output_boxes = []
        output_labels = []

        for box in boxes:
            # box coordinate from 1. not 0.
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]

            width = (xmax - xmin)+1
            height = (ymax - ymin)+1

            if width < 0 or height< 0:
                print("negative width/height")
                continue


            for iter_crop in range(0, self.num_crops, 1):
                rand_idx = rand_indices[iter_crop]

                margin = np.random.randint(16, 128, size=1)

                if rand_idx == 0:
                    offset_x = xmin-1-margin[0]
                    offset_y = ymin-1-margin[0]
                    crop_maxx = offset_x + target_img_size
                    crop_maxy = offset_y + target_img_size

                    if crop_maxx > ori_img_size-1 or crop_maxy > ori_img_size-1:
                        continue
                    if offset_x < 0 or offset_y < 0:
                        continue

                    crop_rect = [offset_x, offset_y, target_img_size, target_img_size]

                    in_boxes, in_labels = self.find_boxes_in_crop(crop_rect, boxes, labels)

                    if in_boxes.__len__() == 0:
                        continue

                    output_offsets.append([offset_x, offset_y])
                    output_boxes.append(in_boxes)
                    output_labels.append(in_labels)

                elif rand_idx == 1:
                    offset_x = xmin - (target_img_size - width)-1+margin[0]
                    offset_y = ymin-1-margin[0]
                    crop_maxx = offset_x + target_img_size
                    crop_maxy = offset_y + target_img_size

                    if crop_maxx > ori_img_size-1 or crop_maxy > ori_img_size-1:
                        continue

                    if offset_x < 0 or offset_y < 0:
                        continue

                    crop_rect = [offset_x, offset_y, target_img_size, target_img_size]

                    in_boxes, in_labels = self.find_boxes_in_crop(crop_rect, boxes, labels)

                    if in_boxes.__len__() == 0:
                        continue

                    output_offsets.append([offset_x, offset_y])
                    output_boxes.append(in_boxes)
                    output_labels.append(in_labels)

                elif rand_idx == 2:
                    offset_x = xmin-1-margin[0]
                    offset_y = ymin - (target_img_size - height)-1+margin[0]
                    crop_maxx = offset_x + target_img_size
                    crop_maxy = offset_y + target_img_size

                    if crop_maxx > ori_img_size-1 or crop_maxy > ori_img_size-1:
                        continue

                    if offset_x < 0 or offset_y < 0:
                        continue

                    crop_rect = [offset_x, offset_y, target_img_size, target_img_size]

                    in_boxes, in_labels = self.find_boxes_in_crop(crop_rect, boxes, labels)

                    if in_boxes.__len__() == 0:
                        continue

                    output_offsets.append([offset_x, offset_y])
                    output_boxes.append(in_boxes)
                    output_labels.append(in_labels)

                elif rand_idx == 3:
                    offset_x = xmin - (target_img_size - width)-1+margin[0]
                    offset_y = ymin - (target_img_size - height)-1+margin[0]
                    crop_maxx = offset_x + target_img_size
                    crop_maxy = offset_y + target_img_size

                    if crop_maxx > ori_img_size-1 or crop_maxy > ori_img_size-1:
                        continue

                    if offset_x < 0 or offset_y < 0:
                        continue

                    crop_rect = [offset_x, offset_y, target_img_size, target_img_size]

                    in_boxes, in_labels = self.find_boxes_in_crop(crop_rect, boxes, labels)

                    if in_boxes.__len__() == 0:
                        continue

                    output_offsets.append([offset_x, offset_y])
                    output_boxes.append(in_boxes)
                    output_labels.append(in_labels)

                elif rand_idx == 4:
                    rand_direction = np.random.randint(-1, 1, size=1)

                    offset_x = (xmin - ((target_img_size-width)/2)-1) + (rand_direction[0] * margin[0])
                    offset_y = (ymin - ((target_img_size-height)/2)-1) + (rand_direction[0] * margin[0])
                    crop_maxx = offset_x + target_img_size
                    crop_maxy = offset_y + target_img_size

                    if crop_maxx > ori_img_size-1 or crop_maxy > ori_img_size-1:
                        continue

                    if offset_x < 0 or offset_y < 0:
                        continue

                    crop_rect = [offset_x, offset_y, target_img_size, target_img_size]

                    in_boxes, in_labels = self.find_boxes_in_crop(crop_rect, boxes, labels)

                    if in_boxes.__len__() == 0:
                        continue

                    output_offsets.append([offset_x, offset_y])
                    output_boxes.append(in_boxes)
                    output_labels.append(in_labels)

                else:
                    print("exceed possible crop num")

        return output_offsets, output_boxes, output_labels


    def find_boxes_in_crop(self, crop_rect, boxes, labels):
        num_boxes = boxes.__len__()
        num_labels = labels.__len__()

        if num_boxes != num_labels:
            print("error occur: Random crop")

        boxes_in_crop=[]
        labels_in_crop = []
        for idx in range(0, num_boxes, 1):
            box_in_crop, label, is_contain = self.find_box_in_crop(crop_rect, boxes[idx], labels[idx])

            if is_contain is True:
                boxes_in_crop.append(box_in_crop)
                labels_in_crop.append(label)

        return boxes_in_crop, labels_in_crop

    def find_box_in_crop(self, rect, box, label):
        rect_minx = rect[0]
        rect_miny = rect[1]
        rect_width = rect[2]
        rect_height = rect[3]

        box_minx = box[0]
        box_miny = box[1]
        box_maxx = box[2]
        box_maxy = box[3]
        box_width = (box_maxx - box_minx)+1
        box_height = (box_maxy - box_miny)+1

        occlusion_ratio = 0.3
        occlusion_width = int(box_width * occlusion_ratio) * -1
        occlusion_height = int(box_height * occlusion_ratio) * -1

        box_in_crop_minx = box_minx - rect_minx
        if box_in_crop_minx <= occlusion_width or box_in_crop_minx >= rect_width:
            box_in_rect = []
            return box_in_rect, label, False

        box_in_crop_miny = box_miny - rect_miny
        if box_in_crop_miny <= occlusion_height or box_in_crop_miny >= rect_height:
            box_in_rect = []
            return box_in_rect, label, False

        box_in_crop_maxx = box_maxx - rect_minx
        if rect_width - box_in_crop_maxx <= occlusion_width or box_in_crop_maxx <= 0:
            box_in_rect = []
            return box_in_rect, label, False

        box_in_crop_maxy = box_maxy - rect_miny
        if rect_height - box_in_crop_maxy <= occlusion_height or box_in_crop_maxy <= 0:
            box_in_rect = []
            return box_in_rect, label, False

        if box_in_crop_minx < 0:
            box_in_crop_minx = 0
        if box_in_crop_miny < 0:
            box_in_crop_miny = 0
        if rect_width - box_in_crop_maxx < 0:
            box_in_crop_maxx = rect_width-1
        if rect_height - box_in_crop_maxy < 0:
            box_in_crop_maxy = rect_height-1

        box_in_rect = [box_in_crop_minx, box_in_crop_miny, box_in_crop_maxx, box_in_crop_maxy]
        return box_in_rect, label, True

    def random_flip(self, img, boxes):
        '''Randomly flip the image and adjust the boxes.

        For box (xmin, ymin, xmax, ymax), the flipped box is:
        (w-xmax, ymin, w-xmin, ymax).

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) object boxes, sized [#obj, 4].

        Returns:
          img: (PIL.Image) randomly flipped image.
          boxes: (tensor) randomly flipped boxes, sized [#obj, 4].

        Reference:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/utils/blob.py
        '''
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
        return img, boxes

    def scale_jitter(self, img, boxes):
        '''Scale image size randomly to [3/4,4/3].

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) object boxes, sized [#obj, 4].

        Returns:
          img: (PIL.Image) scaled image.
          boxes: (tensor) scaled object boxes, sized [#obj, 4].

        Reference:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/utils/blob.py
        '''
        imw, imh = img.size
        sw = random.uniform(3/4., 4/3.)
        sh = random.uniform(3/4., 4/3.)
        w = int(imw*sw)
        h = int(imh*sh)
        img = img.resize((w,h))
        boxes[:,::2] *= sw
        boxes[:,1::2] *= sh
        return img, boxes

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.

        Reference:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/utils/blob.py
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        masks = [x[3] for x in batch]

        max_h = max([im.size(1) for im in imgs])
        max_w = max([im.size(2) for im in imgs])
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, max_h, max_w)
        mask_targets = torch.zeros(num_imgs, max_h, max_w).type(torch.LongTensor)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            im = imgs[i]
            imh, imw = im.size(1), im.size(2)
            inputs[i,:,:imh,:imw] = im

            # Encode data.
            loc_target, cls_target = self.data_encoder.encode(boxes[i], labels[i], input_size=(max_w,max_h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)

            mask = masks[i]
            mask_targets[i,:imh,:imw] = mask

        return inputs, torch.stack(loc_targets), torch.stack(cls_targets), mask_targets

    def __len__(self):
        return self.num_samples

    def convert_from_name_to_label(self, class_name, label_map):
        label = 0

        for item in label_map:
            if item[0] == class_name:
                label = int(item[1])
                break

        return label


def test():
    import torchvision

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    # ])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ListDataset(root="../train",gt_extension=".txt",
                          labelmap_path="class_label_map.xlsx", is_train=True, transform=transform, input_image_size=512,
                          num_crops=1, original_img_size=1024)
    print(dataset.__len__())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn)

    for images, loc_targets, cls_targets, mask_targets in dataloader:
        print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        print(mask_targets.size())
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid,'a.jpg')
        break

# test()
