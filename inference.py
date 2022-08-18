import torch

# import some common libraries
import numpy as np
import os, json, cv2, random
import warnings
warnings.filterwarnings("ignore")

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from .exceptions import handle_exception
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode, GenericMask
import time
import uuid


def format_output(output, custom_metadata):
	items = []
	
	if "instances" in output.keys():
		pred_classes = output["instances"].pred_classes
		pred_scores = output["instances"].scores.tolist()
		pred_classes = np.array([custom_metadata.thing_classes[i] for i in pred_classes])
		pred_classes_counts = np.unique(pred_classes, return_counts=True)
		
		for i, pred in enumerate(pred_classes_counts[0]):
			obj = {}
			obj["name"] = pred
			obj["count"] = pred_classes_counts[1][i]
			obj["location"] = []
			idxs = np.where(pred_classes == pred)
			
			for idx in idxs[0]:
				generic_mask = GenericMask(np.array(output["instances"].pred_masks[idx], dtype=np.uint8),output["height"], output["width"])
				mask = generic_mask.polygons
				mask = ",".join(str(v) for v in mask[0])
				obj["location"].append({
					"coordinates": mask,
					"accuracy_pct": round(pred_scores[idx]*100)
				})
			items.append(obj)
	output["items"] = items
	return output


def prepare_model(data_repo, model_repo):

	basic_config = f"{data_repo}/basic_config.yaml"
	adv_config = f"{data_repo}/advanced_config.yaml"

	# Load annotations json
	with open(f"{data_repo}/metadata.json") as f:
		categories = json.load(f)

	ids = [category["id"] for category in categories]

	categories = [category["name"] for category in categories]

	#add dummy class if 0 not in categories
	if 0 not in ids:
		categories.insert(0, "N/A")

	#Register metadata
	MetadataCatalog.get("custom_test").set(thing_classes=categories)

	#Get metadata
	custom_metadata = MetadataCatalog.get("custom_test")

	model_to_use = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
	cfg = get_cfg()
	
	#add custom key "INFER_DEBUG" and then merge
	cfg.INFER_DEBUG = None
	
	
	if os.path.exists(adv_config): 
		cfg.merge_from_file(adv_config)
	else:
		print("Merging custom configurations")
		cfg.merge_from_file(basic_config)
	
	cfg.INPUT.MASK_FORMAT="polygon"
	cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
	
	#prediction
	cfg.MODEL.WEIGHTS = f"{model_repo}/model_final.pth"
	
	predictor = DefaultPredictor(cfg)
	
	
	return predictor, cfg.INFER_DEBUG, custom_metadata


def infer(debug_path, predictor, debug, meta, files):

	os.makedirs(debug_path, exist_ok=True)

	for file_ in files:

		# Get file BytesIO object
		byte_img = file_.file._file

		# Converting to NumPy array to make it compatible for cv2 decoding
		file_bytes = np.asarray(bytearray(byte_img.read()), dtype=np.uint8)
		
		output = {}
		output["scale_factor"] = 0.0
		output["image_name"] = ""
		output["height"], output["width"] = 0, 0
		output["error"] = None
		
		try:
			# Loading image through bytes
			im = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
			scale = 1
			
			output = predictor(im)
			if debug:
				v = Visualizer(im[:, :, ::-1],
									     metadata=meta,
									     scale=scale,
									     instance_mode=ColorMode.IMAGE_BW
				)
				
				out = v.draw_instance_predictions(output["instances"].to("cpu"))
				im = out.get_image()[:, :, ::-1]
				
			file_name = str(uuid.uuid4())+".jpg"
			cv2.imwrite(f"{debug_path}/{file_name}", im)
			output["scale_factor"] = scale
			output["image_name"] = file_name
			output["height"], output["width"] = im.shape[:2]

		except Exception as e:
			output["error"] = handle_exception(e)
			return output
			
	output = format_output(output, meta)
	
	return output

