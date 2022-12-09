import json
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch.cuda as cuda
import os
import torch
from pycocotools import mask as m
import numpy as np
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode, Instances, Boxes
import logging
import cv2
import math
import tqdm

logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

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
	num_classes = max(ids)+1 if min(ids)==0 else max(ids)
	
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
	cfg.SOLVER.EVAL_PERIOD = 1000
	cfg.SOLVER.PATIENCE = 3
	cfg.INPUT.INFER_SIZE = 1280

	cfg.MODEL.DEVICE = "cuda" if cuda.is_available() else "cpu"

	cfg.DATASETS.TRAIN = ("custom_train",)
	cfg.DATASETS.TEST = ()
	
	cfg.DATALOADER.NUM_WORKERS = 0
	cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
	cfg.DATALOADER.REPEAT_THRESHOLD = 0.5

	cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
	cfg.MODEL.RPN.IN_FEATURES.insert(0, "p2")
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
	cfg.MODEL.ANCHOR_GENERATOR.SIZES.insert(0, [16])
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_to_use)
	cfg.MODEL.ROI_HEADS.NUM_CLASSES, annotations = get_num_classes(train_data_repo)
	cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
	cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
	cfg.MODEL.BACKBONE.FREEZE_AT = 5
	
	cfg.INPUT.MIN_SIZE_TRAIN = (640, 800, 960, 1280)
	cfg.INPUT.MAX_SIZE_TRAIN = 1333
	cfg.INPUT.MIN_SIZE_TEST = 1280
	cfg.INPUT.MAX_SIZE_TEST = 1333
	cfg.INPUT.CROP.ENABLED = True
	cfg.INPUT.MASK_FORMAT = get_mask_type(annotations)

	cfg.SOLVER.IMS_PER_BATCH = 1
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
				
				#wrfite compressed annotations back to {upload_type}.json
				with open(file_path, "w") as f:
					json.dump(annotations, f)
			

def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')
        

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
  
  
async def tile(data_repo, upload_type):
    return

    img_dir = save_dir = data_repo
	
    with open(f"{img_dir}/{upload_type}.json") as f:
        annotations = json.load(f)
	
    all_images = annotations['images']

    ann_idx = 0
    im_idx = 0
    new_anns = []
    new_images = []
    num_of_splits = 6
    
    for img in tqdm.tqdm(all_images):
        im = cv2.imread(f"{img_dir}/{img['file_name']}")
        height = im.shape[0]
        width = im.shape[1]
        
        sample_annotations = [annotation for annotation in annotations['annotations']
                         if annotation['image_id']==img['id']]
        
        cell_width = width / (num_of_splits//2)
        cell_height = height / (num_of_splits//2)
    
        cell_x_coords = [(i*cell_width) for i in range(0,num_of_splits//2)]
        cell_y_coords = [(i*cell_height) for i in range(0,num_of_splits//2)]
    
        cell_width = int(cell_width)
        cell_height = int(cell_height)
    
        for i in range(1,len(cell_x_coords)):
            cell_x_coords[i] = math.floor(cell_x_coords[i])
            cell_y_coords[i] = math.floor(cell_y_coords[i])
            
        # get each tile and associate new masks
        curr_im_idx = 0
        for i in range(num_of_splits//2):
            for j in range(num_of_splits//2):
    
                current_anns = []
    
                min_x = int(cell_x_coords[j])
                min_y = int(cell_y_coords[i])
                max_x = min_x + cell_width
                max_y = min_y + cell_height
                
                tile = im[min_y:max_y,min_x:max_x,:]
                
                new_img_obj = {
                    'id':im_idx,
                    'file_name': f"{curr_im_idx}_{img['file_name']}",
                    'height': tile.shape[0],
                    'width': tile.shape[1]
                }
    
                new_images.append(new_img_obj)
                
                curr_im_idx += 1 
                
                for ann in sample_annotations:
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape(-1,2).astype(float)
    
                        mask_polygon = Polygon([(x,y) for x,y in poly])
                        image = Polygon([(min_x, min_y),(max_x,min_y),
                                         (max_x, max_y),(min_x,max_y)])
                        
                        
                        if not mask_polygon.is_valid:
                        	mask_polygon = make_valid(mask_polygon)
                        result = (image & mask_polygon)
                        
                        if result and isinstance(result, (Polygon, MultiPolygon)):
                            
                            if isinstance(result,MultiPolygon):
                                result = result.geoms[0]
                                
                            coords = result.exterior.coords.xy
                            xs = coords[0].tolist()[1:]
                            ys = coords[1].tolist()[1:]
    
                            xs = [x-min_x for x in xs]
                            ys = [y-min_y for y in ys]
    
                            poly = [[xs[l],ys[l]] for l in range(len(xs))]
                            poly = [m for n in poly for m in n]
    
                            bbox = result.bounds
                            bbox = [bbox[0]-min_x, bbox[1]-min_y, bbox[2]-min_x, bbox[3]-min_y]
    
                            bbox = list(xyxy_to_xywh(bbox))
                            area = result.area
                            ann_obj = {
                                'id':ann_idx,
                                'image_id':im_idx,
                                'category_id':ann['category_id'],
                                'iscrowd':0,
                                'area': area,
                                'bbox': bbox,
                                'segmentation':[poly]
                            }
    
                            new_anns.append(ann_obj)
                            current_anns.append(ann_obj)
    
                            ann_idx += 1           
                im_idx += 1
                cv2.imwrite(f"{save_dir}/{new_img_obj['file_name']}",tile)
        os.remove(f"{img_dir}/{img['file_name']}")
                
    annotations = {
    'images': new_images,
    'annotations':new_anns,
    'categories': annotations['categories']
    }
    
    with open(f"{img_dir}/{upload_type}.json", 'w') as f:
        json.dump(annotations, f)
        
        
        
def sahi_to_detectron_instances(image, sahi_annotations):
        im_height = image.shape[0]
        im_width = image.shape[1]
        pred_boxes = []
        scores = []
        pred_classes = []
        pred_masks = []

        for ann in sahi_annotations:
            pred_boxes.append(ann.bbox.to_xyxy())
            scores.append(ann.score.value)
            pred_classes.append(ann.category.id)
            pred_masks.append(ann.mask.bool_mask)
            
        detectron_dict = {
            'pred_boxes': Boxes(torch.tensor(pred_boxes)),
            'scores':torch.tensor(scores),
            'pred_classes':torch.tensor(pred_classes),
            'pred_masks': torch.tensor(pred_masks)
        }
        
        instances = Instances([im_height, im_width])
        for k in detectron_dict.keys():
            instances.set(k,detectron_dict[k])
        instances = {
            'instances':instances
        }
        return instances
