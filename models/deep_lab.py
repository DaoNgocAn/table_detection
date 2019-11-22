import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCN, FCNHead


def get_segmentation_model(name, num_classes=2):
    model = torchvision.models.segmentation.__dict__[name]()

    deep_lab_head = DeepLabHead(2048, num_classes)
    aux_classifier = FCNHead(1024, num_classes)

    model.classifier = deep_lab_head
    model.aux_classifier = aux_classifier
    return model
