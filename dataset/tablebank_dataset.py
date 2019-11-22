import os
import numpy as np
import torch

import cv2
from PIL import Image
from torch.utils.data import Dataset
import math

import json
from dataset.morph_lines_detection import remove_lines
import random
class TableBank(Dataset):
    TASK = 'FASTER_CRNN'

    def __init__(self, root='/home/andn/TableBank_data', transforms=None, sources=['Word', 'Latex']):
        # root == '/home/andn/Downloads/TableBank/TableBank_data'
        root = os.path.join(root, 'Detection_data/')
        self.ids = []
        for s in sources:
            json_path = os.path.join(root, s, f'{s}.json')
            image_paths = os.path.join(root, s, 'images')
            self.ids += read_json(json_path, image_paths)

        self.root = root
        self.transforms = transforms

    def __getitem__(self, idx):
        oj = self.ids[idx]
        img_path = oj['file_name']
        img = cv2.imread(img_path)
        if random.random() < 0.5:
            img = remove_lines(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        w, h = img.shape[:2]
        _boxes = oj['segmentation']
        boxes = [[] for _ in range(len(_boxes))]
        for id,_box in enumerate(_boxes):
            boxes[id] += [_box[0], _box[1], _box[4], _box[5]]
        labels = [1 for _ in range(len(boxes))]
        num_objs = len(labels)
        for _box in boxes:
            cv2.rectangle(img, (int(_box[0]), int(_box[1])), (int(_box[2]), int(_box[3])), (255, 0, 0), 3)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        if TableBank.TASK == 'FASTER_CRNN':
            mask = np.zeros((num_objs, w, h))
            for i in range(num_objs):
                box = boxes[i]
                box = list(map(int, box))
                _h, _w = box[2] - box[0], box[3] - box[1]
                # heatmap = gaussian_heatmap(_w, _h)
                # mask[0,  box[1]: box[3], box[0]: box[2],] = heatmap
                mask[i, box[1]: box[3], box[0]: box[2], ] = 1

            masks = torch.as_tensor(mask, dtype=torch.uint8)
        else:
            mask = np.zeros((1, w, h))
            for i in range(num_objs):
                box = boxes[i]
                box = list(map(int, box))
                _h, _w = box[2] - box[0], box[3] - box[1]
                # heatmap = gaussian_heatmap(_w, _h)
                # mask[0,  box[1]: box[3], box[0]: box[2],] = heatmap
                mask[0, box[1]: box[3], box[0]: box[2], ] = 1

            if TableBank.TASK.startswith('MY'):
                masks = torch.as_tensor(mask, dtype=torch.uint8)
            else:
                if self.transforms is not None:
                    img, target = self.transforms(Image.fromarray(img), Image.fromarray(mask.squeeze(0)))
                return img, target

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
        target["masks"] = masks
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


def read_json(json_path, image_paths):
    with open(json_path) as f:
        oj = json.load(f)
    annotations = oj['annotations']
    images = oj['images']
    for i in range(len(images)):
        images[i]['file_name'] = os.path.join(image_paths, images[i]['file_name'])
        images[i]['segmentation'] = []
        images[i]['bbox'] = []

    for a in annotations:
        images[a['image_id'] - 1]['segmentation'].append(a['segmentation'][0])
        images[a['image_id'] - 1]['bbox'].append(a['bbox'])

    return images


if __name__ == '__main__':
    dataset = TableBank('/home/andn/Downloads/TableBank/TableBank_data', None, sources=['Latex'])
    for i in dataset:
        pass
