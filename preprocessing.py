import cv2
import os
import numpy


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


if __name__ == '__main__':
    root_path = '/data4/chezhihao/remote_seg_data/Massachusetts Building Dataset/tiff/'
    txt_path = '/data4/chezhihao/remote_seg_data/Massachusetts Building Dataset/VOC2007/ImageSets/Segmentation/'
    save_image_path = '/data4/chezhihao/remote_seg_data/Massachusetts Building Dataset/VOC2007/JPEGImages/'
    save_label_path = '/data4/chezhihao/remote_seg_data/Massachusetts Building Dataset/VOC2007/SegmentationClass/'

    # 分割大小宽和高
    split_width = 500
    split_height = 500

    # 分割图片参数
    name = 'frame'
    frmt = 'jpg'
    lbmt = 'png'

    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)
    if not os.path.exists(save_label_path):
        os.makedirs(save_label_path)

    count = 0
    for sub_path in os.listdir(root_path):
        if sub_path == 'train':
            train_path = root_path + sub_path + '/'
            for img_name in os.listdir(train_path):
                img = cv2.imread(train_path+img_name)
                img_h, img_w, _ = img.shape
                X_points = start_points(img_w, split_width, 0)
                Y_points = start_points(img_h, split_height, 0)

                for i in Y_points:
                    for j in X_points:
                        split = img[i:i + split_height, j:j + split_width]
                        cv2.imwrite(save_image_path + '{}_{}.{}'.format(name, count, frmt), split)
                        with open(txt_path + 'train.txt', 'a') as f:
                            f.write('{}_{}\n'.format(name, count))
                        count += 1

        if sub_path == 'train_labels':
            count = 0
            train_path = root_path + sub_path + '/'
            for img_name in os.listdir(train_path):
                img = cv2.imread(train_path+img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_TRUNC)
                img_h, img_w = img.shape
                X_points = start_points(img_w, split_width, 0)
                Y_points = start_points(img_h, split_height, 0)

                for i in Y_points:
                    for j in X_points:
                        split = img[i:i + split_height, j:j + split_width]
                        cv2.imwrite(save_label_path + '{}_{}.{}'.format(name, count, lbmt), split)
                        count += 1

        count_continue = count
        if sub_path == 'val':
            train_path = root_path + sub_path + '/'
            for img_name in os.listdir(train_path):
                img = cv2.imread(train_path+img_name)
                img_h, img_w, _ = img.shape
                X_points = start_points(img_w, split_width, 0)
                Y_points = start_points(img_h, split_height, 0)

                for i in Y_points:
                    for j in X_points:
                        split = img[i:i + split_height, j:j + split_width]
                        cv2.imwrite(save_image_path + '{}_{}.{}'.format(name, count, frmt), split)
                        with open(txt_path + 'val.txt', 'a') as f:
                            f.write('{}_{}\n'.format(name, count))
                        count += 1

        if sub_path == 'val_labels':
            count = count_continue
            train_path = root_path + sub_path + '/'
            for img_name in os.listdir(train_path):
                img = cv2.imread(train_path + img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_TRUNC)
                img_h, img_w = img.shape
                X_points = start_points(img_w, split_width, 0)
                Y_points = start_points(img_h, split_height, 0)

                for i in Y_points:
                    for j in X_points:
                        split = img[i:i + split_height, j:j + split_width]
                        cv2.imwrite(save_label_path + '{}_{}.{}'.format(name, count, lbmt), split)
                        count += 1

