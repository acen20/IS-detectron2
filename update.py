import yaml, os, shutil
from .utils import populate_default_cfg

def update_config(
	data_repo,
	train_data_repo,
	model_repo,
	sampler_train, 
	repeat_threshold, 
	min_size_train, 
	min_size_test,
	max_size_train, 
	max_size_test, 
	crop_enabled,
	batch_size,
	batch_size_per_image,
	infer_debug,
	checkpoint_period,
	detection_threshold
):

	adv_path = f"{data_repo}/advanced_config.yaml"
	basic_path = f"{data_repo}/basic_config.yaml"
	
	## if basic config does not exist, create default file
	if os.path.exists(basic_path) == False:
		populate_default_cfg(data_repo, train_data_repo, model_repo)

	## if advanced config does not exist, copy basic config 
	## and create a file for advanced config
	if os.path.exists(adv_path) == False:
		shutil.copyfile(basic_path, adv_path)
	
	# create backup
	shutil.copyfile(adv_path, data_repo+'/old_advanced_config.yaml')
	
	with open(adv_path) as f:
		cfg = yaml.load(f)


	# updating the data where the key is not None!
	cfg["DATALOADER"]["SAMPLER_TRAIN"] = sampler_train if sampler_train is not None else cfg["DATALOADER"]["SAMPLER_TRAIN"]
	cfg["DATALOADER"]["REPEAT_THRESHOLD"] = repeat_threshold if repeat_threshold is not None else cfg["DATALOADER"]["REPEAT_THRESHOLD"]
	cfg["INPUT"]["MIN_SIZE_TRAIN"] = (min_size_train,) if min_size_train is not None else cfg["INPUT"]["MIN_SIZE_TRAIN"]
	cfg["INPUT"]["MIN_SIZE_TEST"] = min_size_test if min_size_test is not None else cfg["INPUT"]["MIN_SIZE_TEST"]
	cfg["INPUT"]["MAX_SIZE_TRAIN"] = max_size_train if max_size_train is not None else cfg["INPUT"]["MAX_SIZE_TRAIN"]
	cfg["INPUT"]["MAX_SIZE_TEST"] = max_size_test if max_size_test is not None else cfg["INPUT"]["MAX_SIZE_TEST"]
	cfg["INPUT"]["CROP"]["ENABLED"] = crop_enabled if crop_enabled is not None else cfg["INPUT"]["CROP"]["ENABLED"]
	cfg["SOLVER"]["IMS_PER_BATCH"] = batch_size if batch_size is not None else cfg["SOLVER"]["IMS_PER_BATCH"]
	cfg["MODEL"]["RPN"]["BATCH_SIZE_PER_IMAGE"] = batch_size_per_image if batch_size_per_image is not None else cfg["MODEL"]["RPN"]["BATCH_SIZE_PER_IMAGE"]
	cfg["INFER_DEBUG"] = infer_debug if infer_debug is not None else cfg["INFER_DEBUG"]
	cfg["SOLVER"]["CHECKPOINT_PERIOD"] = checkpoint_period if checkpoint_period is not None else cfg["SOLVER"]["CHECKPOINT_PERIOD"]
	cfg["MODEL"]["ROI_HEADS"]["SCORE_THRESH_TEST"] = detection_threshold if detection_threshold is not None else cfg["MODEL"]["ROI_HEADS"]["SCORE_THRESH_TEST"]
	
	with open(adv_path, 'w') as f:
		yaml.dump(cfg, f)
		
		
