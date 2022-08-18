import torch, detectron2

# import some common libraries
import numpy as np
import os, json, cv2, random
import warnings
warnings.filterwarnings("ignore")

# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.engine.hooks import EvalHook
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from .utils import get_num_classes, populate_default_cfg
from .exceptions import handle_exception
	
	
## Register dataset	
def register_data(data_repo, train_data_repo, test_data_repo):
	MetadataCatalog.clear() 
	DatasetCatalog.clear()

	data_name = "custom"
	train_data_annotations = f"{train_data_repo}/train.json"
	test_data_annotations = f"{test_data_repo}/test.json"

	# Load train annotations json
	with open(f"{train_data_repo}/train.json") as f:
		annotations = json.load(f)

	# Load names of classes for metadata
	categories = annotations["categories"]
	ids = [category["id"] for category in categories]

	# Get number of classses
	num_classes = max(ids)+1

	categories = [category["name"] for category in categories]

	# Add dummy class if 0 not in categories
	if 0 not in ids:
		categories.insert(0, "N/A")

	#Register coco dataset
	register_coco_instances(f"{data_name}_train", {}, train_data_annotations, train_data_repo)
	register_coco_instances(f"{data_name}_test", {}, test_data_annotations, test_data_repo)
	
	custom_metadata = MetadataCatalog.get(data_name + "_" + "train").set(thing_classes=categories)
	
	# save metadata
	with open(f"{data_repo}/metadata.json", "w") as f:
		json.dump(annotations["categories"], f)
		
	return num_classes, custom_metadata
	
	


def early_stopping(cfg, trainer):
    
  # Calculate accuracy/AP
  cfg.DATASETS.TEST = ("custom_test",)
  evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=cfg.OUTPUT_DIR)
  results = trainer.test(cfg, trainer.model, evaluators = [evaluator])
  new_AP = results['bbox']['AP']
  
  # If new AP is "nan", it means the model has not learned anything, so we just return to training loop
  if np.isnan(new_AP):
      return
  
  # Get name of last checkpoint
  with open(os.path.join(cfg.OUTPUT_DIR, 'last_checkpoint'), 'r') as f:
      model_file_name = f.readline()
  
  # current stats
  obj = {
      'model_name': model_file_name,
      'AP': new_AP
  }
  
  # check if there is a history of accuracies by checking if the file exists
  file_name = 'last_check_point_acc.json'
  if os.path.exists(os.path.join(cfg.OUTPUT_DIR, file_name)):
      
    # read previous accuracy
    with open (os.path.join(cfg.OUTPUT_DIR, file_name), 'r') as f:
        previous_stats = json.load(f)
    
    # get previous stats
    previous_AP = previous_stats['AP']
    previous_model_file_name = previous_stats['model_name']
    
    # if new accuracy is less than previous accuracy, stop training!!
    if new_AP < previous_AP:
        # rename previous checkpoint to model_final.pth
        os.rename(os.path.join(cfg.OUTPUT_DIR, previous_model_file_name),
                 os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
        
        #remove current checkpoint, we don't need it any longer
        os.remove(os.path.join(cfg.OUTPUT_DIR, model_file_name))
        
        raise Exception(f"Training finished at {trainer.iter}th iteration! Saving Model!")
    
    else: # continue training
        # write current stats
        with open(os.path.join(cfg.OUTPUT_DIR, file_name), 'w') as f:
            json.dump(obj, f)
            
        # remove previous model file
        os.remove(os.path.join(cfg.OUTPUT_DIR, previous_model_file_name))
                  
  else:
      with open(os.path.join(cfg.OUTPUT_DIR, file_name), 'w') as f:
          json.dump(obj, f)


	
def evaluate_trained(cfg, trainer):
	cfg.DATASETS.TEST = ("custom_test",)
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
	predictor = DefaultPredictor(cfg)
	evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=cfg.OUTPUT_DIR)
	results = trainer.test(cfg, predictor.model, evaluators = [evaluator])
	with open(f'{cfg.OUTPUT_DIR}/results.json', 'w') as f:
		json.dump(results, f)
	
	
## start training

def train_save_model(data_repo, train_data_repo, test_data_repo, model_repo, config):
	trainer = None
	basic_config = f"{data_repo}/basic_config.yaml"
	adv_config = f"{data_repo}/advanced_config.yaml"
	cfg = get_cfg()
	
	# Register training data
	num_classes, custom_metadata = register_data(data_repo, train_data_repo, test_data_repo)
	
	#If no config exists, first write the basic config to directory
	if os.path.exists(basic_config) == False:	
		#get default cfg values
		cfg = populate_default_cfg(data_repo, train_data_repo, model_repo)
		# the number of classes and output directory will be set here

	# Define custom keys
	cfg.INFER_DEBUG = None	
		
	if config == "advanced":
		if os.path.exists(adv_config):
			cfg.merge_from_file(adv_config)
		else:
			print("Advanced configs not found, merging defaults!")
			cfg.merge_from_file(basic_config)
	else:
		cfg.merge_from_file(basic_config)
	
	os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
	trainer = DefaultTrainer(cfg)
	trainer.resume_or_load(resume=False)
	
	response = None
	
	trainer.register_hooks([EvalHook(cfg.SOLVER.CHECKPOINT_PERIOD, 
                         lambda:early_stopping(cfg, trainer))])
	try:
		trainer.train()
	except Exception as e:
		response = handle_exception(e)
		
	try:	
		evaluate_trained(cfg, trainer)
		print("Evaluation completed! Model saved!")
	except:
		pass
		
	return response

	
