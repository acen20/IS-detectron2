import json
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch.cuda as cuda
import os
from pycocotools import mask as m
import numpy as np


def get_mask_type(annotations):
	mask_type = "polygon"
	annotation = annotations["annotations"][0]["segmentation"]
	if isinstance(annotation, dict):
		mask_type = "bitmask"
	return mask_type

def get_num_classes(train_data_repo):
	# Load train annotations json
	with open(f"{train_data_repo}/train.json") as f:
		annotations = json.load(f)

	# Load names of classes for metadata
	categories = annotations["categories"]
	ids = [category["id"] for category in categories]

	# Get number of classses
	num_classes = max(ids)+1
	
	return num_classes, annotations
	

def populate_default_cfg(data_repo, train_data_repo, model_repo):
	
	basic_config = f"{data_repo}/basic_config.yaml"
	
	# Start training
	model_to_use = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
	
	# Get default configurations
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file(model_to_use))
	
	# Define custom keys for configurations
	cfg.INFER_DEBUG = True

	cfg.MODEL.DEVICE = "cuda" if cuda.is_available() else "cpu"

	cfg.DATASETS.TRAIN = ("custom_train",)
	cfg.DATASETS.TEST = ()
	
	cfg.DATALOADER.NUM_WORKERS = 0
	cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
	cfg.DATALOADER.REPEAT_THRESHOLD = 0.5

	cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
	cfg.MODEL.RPN.IN_FEATURES.insert(0, "p2")
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
	cfg.MODEL.ANCHOR_GENERATOR.SIZES.insert(0, [16])
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_to_use)
	cfg.MODEL.ROI_HEADS.NUM_CLASSES, annotations = get_num_classes(train_data_repo)

	cfg.INPUT.MIN_SIZE_TRAIN = (800, 960, 1280)
	cfg.INPUT.MIN_SIZE_TEST = 1280
	cfg.INPUT.CROP.ENABLED = True
	cfg.INPUT.MASK_FORMAT = get_mask_type(annotations)

	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.BASE_LR = 0.0025
	
	cfg.TEST.DETECTIONS_PER_IMAGE = 1000
	
	cfg.OUTPUT_DIR = model_repo
	
	
	# Write default values to config file
	with open(basic_config, "w") as f:
		f.write(cfg.dump())
	
	return cfg


async def compress_annotations(train_data_repo, upload_type):

	file_path = os.path.join(train_data_repo, upload_type+'.json')
	with open(file_path, "r") as f:
		annotations = json.load(f)
		
		# get sample annotation from train file to check if the data is polygon or RLE
		sample = annotations["annotations"][0]["segmentation"]
		if isinstance(sample, dict): # is RLE?
			if isinstance(sample["counts"], list): # Uncompressed RLE
				for idx,_ in enumerate(annotations["annotations"]):
					annotation = annotations["annotations"][idx]
					seg_counts = annotation["segmentation"]["counts"]
					h, w = tuple(annotation["segmentation"]["size"])
					annotations["annotations"][idx]["segmentation"] = to_encoded_rle(seg_counts, h, w)
				
				#write compressed annotations back to {upload_type}.json
				with open(file_path, "w") as f:
					json.dump(annotations, f)
		
		os.remove(file_path+".lock")
			

def to_encoded_rle(rle_mask, h, w):
  rle_mask = np.array(rle_mask).reshape(-1,2)
  mask = np.zeros((h*w), dtype=np.uint8)
  for (start_idx, steps_) in rle_mask:
      mask[start_idx:start_idx+steps_] = 1.0
  mask = mask.reshape(-1,h).T
  
  encoded_mask = m.encode(np.asfortranarray(mask))
  json_encoded_mask = encoded_mask
  json_encoded_mask["counts"] = str(json_encoded_mask["counts"], encoding="utf-8")
  
  return json_encoded_mask
