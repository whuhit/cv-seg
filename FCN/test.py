import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from evalution_segmentaion import calc_semantic_segmentation_confusion,calc_semantic_segmentation_iou
from dataset import CamvidDataset
from FCN import FCN8s
import cfg

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

BATCH_SIZE = 8
miou_list = [0]

Cam_test = CamvidDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Cam_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

net = FCN8s(12)
net.eval()
net.to(device)
net.load_state_dict(t.load('weights/best.pt', map_location='cpu'))

train_acc = 0
train_miou = 0
train_class_acc = 0
train_mpa = 0
error = 0
confusions = np.zeros((12, 12), dtype=np.int64)

for i, sample in enumerate(test_data):
	data = sample['img'].to(device)
	label = sample['label'].to(device)
	out = net(data)
	out = F.log_softmax(out, dim=1)

	pre_label = out.max(dim=1)[1].data.cpu().numpy()
	pre_label = [i for i in pre_label]

	true_label = label.data.cpu().numpy()
	true_label = [i for i in true_label]

	confusion = calc_semantic_segmentation_confusion(pre_label, true_label)
	confusions += confusion
	print(i)


print("all pixes: ", confusions.sum())

iou = calc_semantic_segmentation_iou(confusions)  # (12, )
pixel_accuracy = np.diag(confusions).sum() / confusions.sum()
class_accuracy = np.diag(confusions) / (np.sum(confusions, axis=1) + 1e-10)

miou = np.nanmean(iou)
mean_class_accuracy = np.nanmean(class_accuracy)

print("miou:", miou)
print("pixel_accuracy:", pixel_accuracy)
print("class_accuracy:", class_accuracy)
print("mean_class_accuracy:", mean_class_accuracy)
