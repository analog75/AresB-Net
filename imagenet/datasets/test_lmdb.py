import lmdb
import torch
import torchvision.transforms as transforms
#from datasets import transforms
from protolmdb import definition_pb2 as pb2
import caffelmdb

image_folder = caffelmdb.ImageFolderLMDB

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
																 std=[0.229, 0.224, 0.225])

val_dataset = image_folder(
		"./data/torch_ilsvrc12_val_lmdb",
		transforms.Compose([
#				transforms.Resize(256),
#				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
		]))

val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=256, shuffle=False,
		num_workers=6, pin_memory=True)

for i, (images, target) in enumerate(val_loader):
#  print("============================================================")
 	print("haha:", i, images.shape, target.shape)
#  print("============================================================")
#}}}
