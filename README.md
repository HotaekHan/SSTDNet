# SSTDNet
Implement 'Single Shot Text Detector with Regional Attention, ICCV 2017 Spotlight' using pytorch.
----------

## This code is work for general object detection problem. not for (oriented) text detection problem.
## I will probably update to handle oriented bounding box as soon as possible :)


[How to use]
1. you need dataset.

  * dataset structure is..<br>
    > /train/0.jpg, /train/0.txt, /valid/0.jpg, /valid/0.txt, ....
  * 0.txt contain position and label of objects like below<br>
    <p>(xmin, ymin, xmax, ymax, label)</p>
    <p>1273.0 935.0 1407.0 1017.0 v1</p>
    <p>911.0 893.0 979.0 953.0 v1</p>
    <p>984.0 889.0 1053.0 948.0 v1</p>
  * To encode label name to integer number, you should define labels in the 'class_lable_map.xlsx"
    <p>v1 1</p>
    <p>v2 2</p>
    <p>....</p>
    * start from 1. not from 0. 0 will be background (in the loss.py).

2. need some settings for dataset reader.
   <p>- see train.py. you can find some code for reading dataset</p>
   <pre><code>
     'trainset = ListDataset(root="../train", gt_extension=".txt", labelmap_path="class_label_map.xlsx", is_train=True, transform=transform, input_image_size=512, num_crops=n_crops, original_img_size=2048)'
     </pre></code>

   - you should set the 'input_image_size' and 'original_img_size'. 'input_image_size' is size of (cropped) image for train. And 'original_img_size' is size of (original) image. I made this parameter to handle high resolution image. if you don't need crop function, -1 for num_crops.


3. Train with your dataset!
        <p>you should define some parameter like learning rate, which optimizer to use, size of batch etc.</p>
