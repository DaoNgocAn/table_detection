import os
import numpy as np
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from dataset.PageReader import PageReader
from dataset.Labels import LABEL_TABLE, \
    LABEL_TABLEBODY, \
    LABEL_TABLECAPTION, \
    LABEL_TABLEFOOTNOTE
from math import exp
import random

class MarnotDataset(Dataset):
    # classes = [LABEL_TABLE, LABEL_TABLEBODY, LABEL_TABLECAPTION, LABEL_TABLEFOOTNOTE]
    classes = [LABEL_TABLEBODY]
    TASK = 'FASTER_CRNN'
    def __init__(self, root='data/data_marmot', transforms=None):
        self.root = root
        self.transforms = transforms
        chinese_negative_ids = [os.path.join(root, 'Chinese', 'Negative', 'Labeled', i[:-4]) for i in
                                os.listdir(os.path.join(root, 'Chinese', 'Negative', 'Labeled'))]  # 500
        chinese_positive_ids = [os.path.join(root, 'Chinese', 'Positive', 'Labeled', i[:-4]) for i in
                                os.listdir(os.path.join(root, 'Chinese', 'Positive', 'Labeled'))]  # 507
        english_negative_ids = [os.path.join(root, 'English', 'Negative', 'Labeled', i[:-4]) for i in
                                os.listdir(os.path.join(root, 'English', 'Negative', 'Labeled'))]  # 484
        english_positive_ids = [os.path.join(root, 'English', 'Positive', 'Labeled', i[:-4]) for i in
                                os.listdir(os.path.join(root, 'English', 'Positive', 'Labeled'))]  # 509
        chinese_negative_ids = []
        english_negative_ids = []
        self.ids = chinese_negative_ids + chinese_positive_ids + english_negative_ids + english_positive_ids
        remove = []
        marmot_c = [i[:-8] for i in os.listdir('marmot_c')]
        for idx in range(len(self.ids)):
            if self.ids[idx].split('/')[-1] not in marmot_c:
                # print(self.ids[idx])
                remove.append(idx)
        # # table
        #
        # table_body
        for i in remove[::-1]:
            del self.ids[i]

        # print(remove)

    def __getitem__(self, idx):
        img_path = self.ids[idx].replace('Labeled', 'Raw') + '.bmp'
        xml_path = self.ids[idx] + '.xml'
        # img_path = '/home/andn/PycharmProjects/table_detection/data/data_marmot/English/Positive/Raw/10.1.1.1.2013_63.bmp'
        # xml_path = '/home/andn/PycharmProjects/table_detection/data/data_marmot/English/Positive/Labeled/10.1.1.1.2013_63.xml'
        page = PageReader.read(xml_path)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # img = Image.open(img_path).convert("RGB")
        w, h = img.shape[:2]
        boxes = []
        labels = []
        for i, label in enumerate(MarnotDataset.classes):
            items = page.getByLabel(label)
            for item in items:
                _box = [item._box._x1, item._box._y1, item._box._x0, item._box._y0]
                _box = list(map(lambda x: x / 72 * 96, _box))
                _box[1] = w - _box[1]
                _box[3] = w - _box[3]
                _box = [_box[2], _box[1], _box[0], _box[3]]

                # _box = list(map(int, _box))
                # cv2.circle(img, (_box[0], _box[1]), 3, (0, 255, 0), -1)
                # cv2.circle(img, (_box[2], _box[3]), 3, (0, 0, 255), -1)
                # print((_box[0], _box[1]), (_box[2], _box[3]))
                # cv2.rectangle(img, (int(_box[0]), int(_box[1])), (int(_box[2]), int(_box[3])), (255, 0, 0), 1)
                # cv2.imshow('image', img)
                # cv2.waitKey(0)

                boxes.append(_box)
                labels.append(i + 1)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        num_objs = len(labels)
        if MarnotDataset.TASK == 'FASTER_CRNN':
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

            if MarnotDataset.TASK.startswith('MY'):
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
    import transforms as T
    import tqdm


    def get_transform(train):
        transforms = []
        # converts the image, a PIL image, into a PyTorch Tensor
        transforms.append(T.ToTensor())
        if train:
            # during training, randomly flip the training images
            # and ground-truth for data augmentation
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)


    dataset = MarnotDataset("/home/andn/PycharmProjects/table_detection/data/data_marmot", None)
    for data_t in tqdm.tqdm(dataset):
        img, tgt = data_t[0], data_t[1]
        # img = img.mul(255).permute(1, 2, 0).byte().numpy()
        boxes = tgt["boxes"].numpy().tolist()
        labels = tgt["labels"].numpy().tolist()
        img_id = tgt["image_id"].numpy().tolist()[0]

        for box, label in zip(boxes, labels):
            box = list(map(int, box))
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 0)
            cv2.putText(img, f'{label}_grou', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255),
                        lineType=cv2.LINE_AA)
        cv2.imwrite(f'/home/andn/PycharmProjects/table_detection/eval_result/{tgt["img_path"]}.jpg', img)