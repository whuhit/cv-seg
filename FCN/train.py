import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import CamvidDataset
from evalution_segmentaion import eval_semantic_segmentation, calc_semantic_segmentation_confusion, calc_semantic_segmentation_iou
from FCN import FCN8s
import cfg
import numpy as np


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

Cam_train = CamvidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Cam_val = CamvidDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(
    Cam_train,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=1)
val_data = DataLoader(
    Cam_val,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=1)

fcn = FCN8s(num_classes=12)
fcn.init("weights/best.pt")
fcn = fcn.to(device)
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(fcn.parameters(), lr=1e-4)


def val(model):
    confusions = np.zeros((12, 12), dtype=np.int64)
    for i, sample in enumerate(val_data):
        data = sample['img'].to(device)
        label = sample['label'].to(device)
        out = model(data)
        out = F.log_softmax(out, dim=1)

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = label.data.cpu().numpy()
        true_label = [i for i in true_label]

        confusion = calc_semantic_segmentation_confusion(pre_label, true_label)
        confusions += confusion
        print(i)
    iou = calc_semantic_segmentation_iou(confusions)  # (12, )
    miou = np.nanmean(iou)
    return miou


def train(model):
    best = 0
    net = model.train()
    # 训练轮次
    for epoch in range(cfg.EPOCH_NUMBER):
        print("Eopch is [{}/{}]".format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5

        train_loss = 0

        for i, sample in enumerate(train_data):
            img_data = sample["img"].to(device)
            img_label = sample["label"].to(device)

            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metric = eval_semantic_segmentation(pre_label, true_label)
            print(f"iteration {i} : miou={eval_metric['miou']}")

        val_miou = val(net)

        if val_miou > best:
            best = val_miou
            torch.save(net.state_dict(), f"weights/best-{val_miou}.pth")
        print(f"epoch:{epoch}, miou:{val_miou}")


train(fcn)
