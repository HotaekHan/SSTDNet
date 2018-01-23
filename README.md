# SSTDNet
Implement 'Single Shot Text Detector with Regional Attention, ICCV 2017 Spotlight' using pytorch.


[How to use]<br>
1. you need dataset<br>
<br>
  - dataset structure is..<br>
    /train/0.jpg<br>
    /train/0.txt<br>
    /valid/0.jpg<br>
    /valid/0.txt<br>
    ....<br>
  - 0.txt contain position and label of object like below<br>
    (xmin, ymin, xmax, ymax, label)<br>
    1273.0 935.0 1407.0 1017.0 v1<br>
    911.0 893.0 979.0 953.0 v1<br>
    984.0 889.0 1053.0 948.0 v1<br>
  - To encode label name to integer number, you define label in the 'class_lable_map.xlsx"<br>
    v1 1<br>
    v2 2<br>
    ....<br>
    * start from 1. not from 0. 0 will be background.<br>

2. some setting for dataset reader.<br>
   - see train.py. you can find some code for reading dataset<br>
     'trainset = ListDataset(root="../train", gt_extension=".txt", labelmap_path="class_label_map.xlsx", is_train=True, transform=transform, input_image_size=512, num_crops=n_crops, original_img_size=2048)'

   - you should set the 'input_image_size' and 'original_img_size'. 'input_image_size' is size of (cropped) image for train. And 'original_img_size' is size of (original) image. I made this parameter to handle high resolution image. if you don't need crop function, -1 for num_crops.<br>
<br>
<br>
3. Train with your dataset!<br>
        you should define some parameter like learning rate, which optimizer to use, size of batch etc.<br>
