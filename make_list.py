import os
from PIL import Image
import numpy as np

image_path_train = '/data4/chezhihao/remote_seg_data/WHU Dataset/train/image/'
image_path_val = '/data4/chezhihao/remote_seg_data/WHU Dataset/val/image/'
save_image_path = '/data4/chezhihao/remote_seg_data/WHU Dataset/VOC2007/JPEGImages/'
save_label_parh = '/data4/chezhihao/remote_seg_data/WHU Dataset/VOC2007/SegmentationClass/'

count = 0
print('--------train part--------')
for img in os.listdir(image_path_train):
    img_path = image_path_train + img
    label_path = img_path.replace('image', 'label')
    image = Image.open(img_path)
    label = Image.open(label_path).convert('L')
    label = np.array(label)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 255:
                label[i][j] = 1
    save_path = save_image_path + str(count) + '.jpg'
    image.save(save_path)
    print(save_path)
    save_path = save_label_parh + str(count) + '.png'
    label = Image.fromarray(label)
    label.save(save_path)
    print(save_path)
    count += 1

print('--------val part--------')
for img in os.listdir(image_path_val):
    img_path = image_path_val + img
    label_path = img_path.replace('image', 'label')
    image = Image.open(img_path)
    label = Image.open(label_path).convert('L')
    label = np.array(label)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 255:
                label[i][j] = 1
    save_path = save_image_path + str(count) + '.jpg'
    image.save(save_path)
    print(save_path)
    save_path = save_label_parh + str(count) + '.png'
    label = Image.fromarray(label)
    label.save(save_path)
    print(save_path)
    count += 1
