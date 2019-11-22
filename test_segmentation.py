import glob

import numpy as np
import cv2
import torch
import torchvision
import tqdm
from torchvision.transforms import functional as F
from torch.nn.functional import softmax
# from models.segmentation import get_segmentation_model
from models.deep_lab import get_segmentation_model

model = torchvision.models.segmentation.__dict__['fcn_resnet101'](num_classes=2,
                                                                  aux_loss=False,
                                                                  pretrained=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
model.to(device)

checkpoint = torch.load('seg_model/model_5.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

for img_path in tqdm.tqdm(glob.glob("test/*")):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = F.to_tensor(img)
    img = img[None, ]
    with torch.no_grad():
        pred = model(img.to(device))
        pred = softmax(pred['out'], dim=1)
        mask = pred.cpu().data.numpy().squeeze()[0]
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0
    mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
    cv2.imwrite('test_result/' + img_path.split('/')[-1][:-4] + '_mask.png', mask.astype(np.uint8))
    # cv2.imshow('name', mask)

    # cv2.waitKey(0)

    # predict_boxes = prediction[0]['boxes'].cpu().numpy().tolist()
    # predict_labels = prediction[0]['labels'].cpu().numpy().tolist()
    # img = img.mul(255).permute(1, 2, 0).byte().numpy()
    #
    # for box, label in zip(predict_boxes, predict_labels):
    #     box = list(map(int, box))
    #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
    #     cv2.putText(img, f'{label}_pred', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255),
    #                 lineType=cv2.LINE_AA)
    # name = img_path.split('/')[-1]
    # cv2.imwrite(f'test_result/{name}', img)
