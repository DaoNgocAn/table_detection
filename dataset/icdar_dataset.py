import os
import random
from math import exp
import xml.etree.ElementTree as ET

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from dataset.utils import Table1, Table2


class ICDAR(Dataset):
    TASK = 'FASTER_CRNN'

    def __init__(self, root='data/data_icdar', transforms=None, mode='train'):
        self.root = root
        self.transforms = transforms
        if mode == 'all':
            self.ids = []
            for m in ['train', 'test']:
                self.ids += [os.path.join(root, m, 'TRACKA', i)[:-4] for i in
                             os.listdir(os.path.join(root, m, 'TRACKA')) if i.endswith('xml')]
        else:
            self.ids = [os.path.join(root, mode, 'TRACKA', i)[:-4] for i in
                        os.listdir(os.path.join(root, mode, 'TRACKA')) if i.endswith('xml')]

    def __getitem__(self, idx):
        img_path = self.ids[idx] + '.jpg'
        xml_path = self.ids[idx] + '.xml'
        boxes = get_tables(xml_path)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        w, h = img.shape[:2]
        labels = [1 for _ in range(len(boxes))]
        num_objs = len(labels)
        # < --------------------- TRANSFORMER replace tables with generated tables ------------------------>
        for i, _box in enumerate(boxes):
            _box = list(map(int, _box))
            if random.random() > 0.75 or _box[2] - _box[0] < 450:
                continue
            Table = random.choice([Table1, Table2])
            new_table = Table(table_height=_box[3] - _box[1], table_widths=_box[2] - _box[0])
            alt_img = new_table.draw()
            if new_table.n_rows >= 4:
                w, h = alt_img.shape[:2]
                if img[_box[1]:_box[1] + w, _box[0]:_box[0] + h].shape == alt_img.shape:
                    img[_box[1]:_box[3], _box[0]:_box[2]] = 255
                    img[_box[1]:_box[1] + w, _box[0]:_box[0] + h] = alt_img
                    w, h = new_table.get_table_bounding_box()
                    boxes[i] = [_box[0], _box[1], _box[0] + h, _box[1] + w]

        # <---------------------- \TRANSFORMER ----------------------->

        # <---------------------- DEBUG ----------------------->
        if "PycharmProjects" in os.getcwd():
            for i, _box in enumerate(boxes):
                cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
                              (int(boxes[i][2]), int(boxes[i][3])), (255, 0, 0), 1)
            cv2.imshow('image', img)
            cv2.waitKey(0)
        # <---------------------- DEBUG ----------------------->
        if ICDAR.TASK == 'FASTER_CRNN':
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
            mask = np.zeros((1, w, h), dtype=np.uint8)
            for i in range(num_objs):
                box = boxes[i]
                box = list(map(int, box))
                _h, _w = box[2] - box[0], box[3] - box[1]
                # heatmap = gaussian_heatmap(_w, _h)
                # mask[0,  box[1]: box[3], box[0]: box[2],] = heatmap
                mask[0, box[1]: box[3], box[0]: box[2], ] = 1
            if ICDAR.TASK.startswith('MY'):
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


def get_tables(xml_file):
    root = ET.parse(xml_file).getroot()
    bboxes = []
    for element in root.findall("table/Coords"):
        points = element.attrib['points'].split()
        box = points[0].split(',') + points[2].split(',')
        box = list(map(int, box))
        bboxes.append(box)
    return bboxes


def scaledGaussian(x):
    return exp(-(1 / 2) * (x ** 2))


def gaussian_heatmap(h, w):
    isotropicGrayscaleImage = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            # find euclidian distance from center of image (imgSize/2,25imgSize/25)
            # and scale it to range of 0 to 2.5 as scaled Gaussian
            # returns highest probability for x=0 and approximately
            # zero probability for x > 2.5
            distanceFromCenter = 1.5 * np.linalg.norm(np.array([i / h - 1 / 2, j / w - 1 / 2])) / (1 / 2)
            scaledGaussianProb = scaledGaussian(distanceFromCenter)
            isotropicGrayscaleImage[i, j] = scaledGaussianProb

    return isotropicGrayscaleImage


if __name__ == '__main__':
    from dataset.marnot_dataset import MarnotDataset

    root = '/home/andn/PycharmProjects/table_detection/data/data_icdar'
    mode = 'all'
    dataset_icdar = ICDAR(root, None, mode)
    dataset = dataset_icdar
    print(len(dataset))
    print(len(dataset_icdar))

    for i in dataset:
        pass
    # get_tables('/home/andn/PycharmProjects/table_detection/data/data_icdar/train/TRACKA/cTDaR_t10003.xml')
