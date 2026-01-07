from exp.exp_main import Exp_Main
from utils.globals import logger, accelerator
import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import datetime
from utils.metrics import metric

class Exp_Robustness(Exp_Main):
    def drop_input_points(self, batch, drop_rate):
        """
        batch: dict containing 'x', 'x_mask', etc.
        drop_rate: float (e.g., 0.3 for 30% drop)
        """
        if drop_rate == 0:
            return batch
        
        x = batch['x']
        x_mask = batch['x_mask']
        
        # Generate random mask for dropping
        # We want to drop with probability 'drop_rate'
        # So we keep with probability '1 - drop_rate'
        keep_mask = torch.rand_like(x_mask) > drop_rate
        
        # New mask is: was_observed AND kept
        # x_mask is 1 for observed, 0 for missing
        new_x_mask = x_mask * keep_mask.float()
        
        # Update x: set to 0 where we dropped
        # We assume x is already 0 where x_mask is 0 (standard in this repo)
        # So we just need to zero out the newly dropped points
        new_x = x * keep_mask.float()
        
        batch['x'] = new_x
        batch['x_mask'] = new_x_mask
        
        return batch

    def test_robustness(self, drop_rates=[0.1, 0.2, 0.3, 0.4, 0.5]):
        logger.info(f'>>>>>>> Robustness testing start (Drop rates: {drop_rates}) <<<<<<<')

        # test_all will test the model on all available sets (train, val, test). Needs to be supported by the dataset
        # Usually for robustness we only care about the test set
        flag = "test" 
        test_data, test_loader = self._get_data(flag=flag)

        # find model checkpoint path
        checkpoint_location: Path = None
        actual_itrs = 1
        if self.configs.checkpoints_test is None:
            # by default, if checkpoints_test is not given, it tries to load the latest corresponding checkpoint
            checkpoint_location = Path(self.configs.checkpoints) / self.configs.dataset_name / self.configs.model_name / self.configs.model_id / f"{self.configs.seq_len}_{self.configs.pred_len}"
            if self.configs.load_checkpoints_test:
                try:
                    # first, find the latest one based on timestamp in name
                    child_folders = [(entry.name, entry) for entry in checkpoint_location.iterdir() if entry.is_dir()]
                    if len(child_folders) == 0:
                        logger.exception(f"No folder under '{checkpoint_location}' matches the model_id '{self.configs.model_id}'.", stack_info=True)
                        logger.exception(f"Tips: Failed to infer the latest checkpoint folder. Please manually provide the checkpoints_test argument pointing to the folder of checkpoint file")
                        exit(1)
                    latest_folder: str = sorted(child_folders, key=lambda item: datetime.datetime.strptime(item[0], "%Y_%m%d_%H%M"))[-1][1].name
                    checkpoint_location = checkpoint_location / latest_folder
                    # then find the latest iter
                    actual_itrs = len([entry.name for entry in checkpoint_location.iterdir() if entry.is_dir()])
                except Exception as e:
                    logger.exception(f"{e}", stack_info=True)
                    logger.exception(f"Tips: Failed to infer the latest checkpoint folder. Please manually provide the checkpoints_test argument pointing to the folder of checkpoint file")
                    exit(1)
            else:
                # If not loading checkpoints, we can't do robustness test on trained models
                logger.error("Robustness test requires loading checkpoints. Please set load_checkpoints_test=1.")
                exit(1)
        
        # Create a parent folder for this robustness run
        robustness_folder_name = f'robustness_eval_{datetime.datetime.now().strftime("%Y_%m%d_%H%M")}'
        robustness_root_path = checkpoint_location / robustness_folder_name
        robustness_root_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Robustness results will be saved under {robustness_root_path}")

        # test on all iters' checkpoints
        for itr_i in range(actual_itrs):
            if self.configs.checkpoints_test is None:
                checkpoint_location_itr = checkpoint_location / f"iter{itr_i}"
            else:
                checkpoint_location_itr = Path(self.configs.checkpoints_test)

            logger.info(f"Loading checkpoint from {checkpoint_location_itr}")
            
            model_test = self._build_model().eval()
            # load model checkpoint
            checkpoint_file = checkpoint_location_itr / "pytorch_model.bin"
            if checkpoint_file.exists():
                try: 
                    original_state_dict = self._get_state_dict(checkpoint_file)
                    load_result = model_test.load_state_dict(original_state_dict, strict=False)
                    if load_result.missing_keys or load_result.unexpected_keys:
                        logger.warning(f"Checkpoint loading warning: {load_result}")
                except Exception as e:
                    logger.exception(f"{e}", stack_info=True)
                    continue
            else:
                logger.error(f"Checkpoint file not found: {checkpoint_file}")
                continue

            model_test, test_loader_acc = accelerator.prepare(model_test, test_loader)
            if not self.configs.use_multi_gpu:
                model_test = model_test.to(f"cuda:{self.configs.gpu_id}")

            # Iterate over drop rates
            for r in drop_rates:
                logger.info(f"Testing with drop rate: {r} (Iter {itr_i})")
                
                # Create folder for this drop rate
                folder_path = robustness_root_path / f"iter{itr_i}" / f"drop_{int(r*100)}"
                folder_path.mkdir(parents=True, exist_ok=True)

                # dictionary holding input and output data
                array_dict = {}
                input_tensor_names = ["x", "y", "x_mask", "y_mask", "sample_ID"]
                output_tensor_names = ["pred"]

                for tensor_name in input_tensor_names + output_tensor_names:
                    array_dict[tensor_name] = []
                
                with torch.no_grad():
                    batch: dict[str, Tensor] # type hints
                    for i, batch in tqdm(enumerate(test_loader_acc), total=len(test_loader_acc), leave=False, desc=f"Testing Drop {r}"):
                        if not self.configs.use_multi_gpu:
                            batch = {k: v.to(f"cuda:{self.configs.gpu_id}") for k, v in batch.items()}

                        # Apply masking
                        batch = self.drop_input_points(batch, r)

                        outputs: dict[str, Tensor] = model_test(
                            exp_stage="test",
                            **batch
                        )

                        batch_all: list[dict] = accelerator.gather_for_metrics([batch])
                        batch_all: dict = self._merge_gathered_dicts(batch_all)
                        outputs_all: list[dict] = accelerator.gather_for_metrics([outputs])
                        outputs_all: dict = self._merge_gathered_dicts(outputs_all)

                        for tensor_name in input_tensor_names:
                            if tensor_name in batch_all.keys():
                                array_dict[tensor_name].append(batch_all[tensor_name].detach().cpu().numpy())
                        for tensor_name in output_tensor_names:
                            if tensor_name in outputs_all.keys():
                                array_dict[tensor_name].append(outputs_all[tensor_name].detach().cpu().numpy())

                for tensor_name in input_tensor_names + output_tensor_names:
                    if len(array_dict[tensor_name]) > 0:
                        array_dict[tensor_name] = np.concatenate(array_dict[tensor_name], axis=0)
                    else:
                        array_dict[tensor_name] = None

                metrics = metric(**array_dict)
                
                # Save metrics
                if metrics is not None:
                    for key, value in metrics.items():
                        if isinstance(value, np.float32):
                            metrics[key] = float(value)
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, np.float32):
                                    metrics[key] = [float(v) for v in value]
                                    break
                    
                    with open(folder_path / "metric.json", "w") as f:
                        json.dump(metrics, f, indent=2)
                    
                    logger.info(f"Iter {itr_i}, Drop {r}: MSE={metrics.get('MSE', 'N/A')}")

