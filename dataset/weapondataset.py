import os
import numpy as np
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
import glob


class WeaponDataset(Dataset):
    TASK = 'FASTER_RCNN'

    def __init__(self, root='gun_knife/dataset/', transforms=None):
        self.root = root
        self.transforms = transforms
        self.ids = [path[:-4] for path in glob.glob(root + '*') if path.endswith('.jpg')]

    def __getitem__(self, idx):
        img_path = self.ids[idx] + '.jpg'
        ann_path = self.ids[idx] + '.txt'
        img = cv2.imread(img_path)

        lines = read_lines(ann_path)
        boxes = []
        labels = []
        _h, _w = img.shape[:2]
        for line in lines:
            a = line.split()
            labels.append(int(a[0]) + 1)
            x, y, w, h = map(float, a[1:])
            x -= w / 2
            y -= h / 2
            x = int(x * _w)
            w = int(w * _w)
            y = int(y * _h)
            h = int(h * _h)
            boxes.append((x, y, x + w, y + h))
        # for _box in boxes:
        #     cv2.rectangle(img, (int(_box[0]), int(_box[1])), (int(_box[2]), int(_box[3])), (255, 0, 0), 1)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        num_objs = len(labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        # target["img_path"] = img_path.split('/')[-1]
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


def read_lines(file_path):
    with open(file_path) as f:
        return f.readlines()


if __name__ == '__main__':
    dataset = WeaponDataset('/home/andn/gun_knife/dataset/')
    for _ in dataset:
        pass
