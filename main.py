import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

import matplotlib.pyplot as plt
from PIL import Image


def localization(path):
    """
        Выполняет локализацию объекта на изображении с помощью CAM (Class Activation Map).

        Аргументы:
            path (str): путь к изображению

        Возвращает:
            img (np.ndarray): восстановленное изображение
            img_numpy (np.ndarray): карта активации
            size (tuple): координаты (x1, y1, x2, y2)
            class_name (str): имя предсказанного класса
        """
    model = models.resnet34(weights='DEFAULT')

    img = Image.open(fr'{path}')

    transforms = models.ResNet34_Weights.DEFAULT.transforms()
    class_name = models.ResNet34_Weights.DEFAULT.meta['categories']

    model.eval()

    img_in = transforms(img).unsqueeze(dim=0)
    pred = model(img_in).squeeze()

    # индекс предсказанного класса indices[0]
    sorted, indices = pred.softmax(dim=0).sort(descending=True)

    new_model = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4
    )

    weights = model.fc.state_dict()['weight'][indices[0]]

    new_model.eval()

    feature_maps = new_model(img_in)

    pred = feature_maps * weights.reshape((feature_maps.shape[1], 1, 1))

    pred = pred.sum(dim=1, keepdims=True)

    img_numpy = nn.Upsample(scale_factor=32, mode='bilinear')(pred).squeeze().detach().numpy()

    mask = img_numpy > img_numpy.max() * 0.2
    img_numpy = img_numpy * mask

    img = transforms(img)
    img = np.transpose(img, axes=[1, 2, 0]) * np.array([[[0.229, 0.224, 0.225]]]) + np.array([[[0.485, 0.456, 0.406]]])

    rows = np.where(img_numpy.any(axis=1))[0]
    cols = np.where(img_numpy.any(axis=0))[0]

    y1, y2 = rows[0], rows[-1]
    x1, x2 = cols[0], cols[-1]

    return img, img_numpy, (x1, y1, x2, y2), class_name[indices[0]]


img, img_numpy, size, name = localization(
    r'C:\Users\STARLINECOMP\PycharmProjects\Pytorch\content\test_set\cats\cat.4001.jpg')

plt.imshow(img_numpy, cmap='gray')
plt.show()

plt.imshow(img)
plt.show()
