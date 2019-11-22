import torch
import cv2
import tqdm
import glob
from torchvision.transforms import functional as F
from models.faster_rcnn import get_faster_crnn_model, get_mask_crnn_model

num_classes = 2
model = get_faster_crnn_model(num_classes)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
model.to(device)

checkpoint = torch.load('model_18.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()



for img_path in tqdm.tqdm(glob.glob("test/*")):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = F.to_tensor(img)
    with torch.no_grad():
        prediction = model([img.to(device)])
    predict_boxes = prediction[0]['boxes'].cpu().numpy().tolist()
    predict_labels = prediction[0]['labels'].cpu().numpy().tolist()
    scores = prediction[0]['scores'].cpu().numpy().tolist()
    img = img.mul(255).permute(1, 2, 0).byte().numpy()
    for box, label, score in zip(predict_boxes, predict_labels, scores):
        box = list(map(int, box))
        if score < 0.1:
            continue
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
        cv2.putText(img, f'score: {score}', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255),
                    lineType=cv2.LINE_AA)
    name = img_path.split('/')[-1]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'test_result/{name}', img)