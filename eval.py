import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from dataset.marnot_dataset import MarnotDataset
import torch
import cv2
import tqdm

num_classes = len(MarnotDataset.classes) + 1

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                   hidden_layer,
                                                   num_classes)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

checkpoint = torch.load('model_21_10/model_60.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


dataset = MarnotDataset('data/data_marmot', get_transform(train=False))
dataset_test = MarnotDataset('data/data_marmot', get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[:2])
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
# define training and validation data loaders

for data_t in tqdm.tqdm(dataset):
    img, tgt = data_t[0], data_t[1]
    with torch.no_grad():
        prediction = model([img.to(device)])
    predict_boxes = prediction[0]['boxes'].cpu().numpy().tolist()
    predict_labels = prediction[0]['labels'].cpu().numpy().tolist()
    img = img.mul(255).permute(1, 2, 0).byte().numpy()
    boxes = tgt["boxes"].numpy().tolist()
    labels = tgt["labels"].numpy().tolist()
    img_id = tgt["image_id"].numpy().tolist()[0]

    for box, label in zip(predict_boxes, predict_labels):
        box = list(map(int, box))
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
        cv2.putText(img, f'{label}_pred', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255),
                    lineType=cv2.LINE_AA)
    # for box, label in zip(boxes, labels):
    #     box = list(map(int, box))
    #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
    #     cv2.putText(img, f'{label}_grou', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255),
    #                 lineType=cv2.LINE_AA)
    cv2.imwrite(f'eval_result/{img_id}.jpg', img)