# modelmodule.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from model_factory import create_model
from collections import OrderedDict
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torchmetrics.functional as TMF
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ModelModule(pl.LightningModule):
    def __init__(self, model_config: dict, stage_config: dict, stage: str):
        super().__init__()
        self.stage = stage
        self.model_config = model_config
        self.stage_config = stage_config

        # --- Configuration for Metrics and CM ---
        self.num_classes = self.model_config.get('d_out')
        if self.num_classes is None:
            raise ValueError("'d_out' (number of classes) must be defined in model config.")
        self.class_names = self.model_config.get('class_names', [f'Class {i}' for i in range(self.num_classes)])
        if len(self.class_names) != self.num_classes:
             print(f"Warning: Number of class_names ({len(self.class_names)}) does not match d_out ({self.num_classes}). Using default names.")
             self.class_names = [f'Class {i}' for i in range(self.num_classes)]

        # Save hyperparameters (includes model config and stage config)
        self.save_hyperparameters({self.stage: stage_config, 'model': model_config})

        # --- Model Initialization ---
        if self.stage == 'distill':
            # Create Student Model
            # Use student_model_config from distill stage config if provided, else use main model_config
            student_config = stage_config.get('student_model_config', None) or model_config
            print("Initializing Student Model...")
            self.student_model = create_model(student_config)
            self.model = self.student_model

            # Create and load Teacher Model
            # Teacher structure comes from the main model_config (must match checkpoint)
            print("Initializing and loading Teacher Model...")
            self.teacher_model = create_model(model_config)
            teacher_ckpt_path = self.stage_config['teacher_ckpt_path']
            print(f"Loading teacher checkpoint from: {teacher_ckpt_path}")
            teacher_checkpoint = torch.load(teacher_ckpt_path, map_location=lambda storage, loc: storage)
            teacher_state_dict = teacher_checkpoint['state_dict']
            teacher_state_dict = {k.replace("model.", ""): v for k, v in teacher_state_dict.items() if k.startswith("model.")}

            self.teacher_model.load_state_dict(teacher_state_dict, strict=True)

            # Freeze teacher model
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            print("Teacher model loaded and frozen.")

            # Distillation parameters
            self.temperature = self.stage_config['temperature']
            self.alpha = self.stage_config['alpha']

        else: # Pretrain, Finetune, Test
            print(f"Initializing model for stage: {self.stage}")
            self.model = create_model(model_config)
            self.teacher_model = None

        # --- Loss Function ---
        self.criterion = nn.CrossEntropyLoss()

        # --- Metrics ---
        task = "multiclass" if self.num_classes > 1 else "binary"
        self.train_acc = Accuracy(task=task, num_classes=self.num_classes, top_k=1)
        self.val_acc = Accuracy(task=task, num_classes=self.num_classes, top_k=1)
        self.test_acc = Accuracy(task=task, num_classes=self.num_classes, top_k=1)

        # --- Initialize lists to store test outputs for Confusion Matrix ---
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, x):
        """Forward pass through the primary model (student in distill stage)."""
        return self.model(x)

    def _calculate_loss(self, batch, stage_prefix):
        """Calculates loss and returns dict with loss, logits, targets."""
        x, y = batch
        logits = self(x)

        hard_loss = self.criterion(logits, y)
        loss = hard_loss

        # --- Distillation Loss Calculation (only during training stage 'distill') ---
        if self.stage == 'distill' and self.teacher_model is not None and stage_prefix=='train':
            self.teacher_model.to(self.device)
            with torch.no_grad():
                teacher_logits = self.teacher_model(x)

            soft_teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
            soft_student_probs = F.log_softmax(logits / self.temperature, dim=1) # Use log_softmax for KLDiv

            kl_div_loss = F.kl_div(soft_student_probs, soft_teacher_probs, reduction='batchmean')
            kl_div_loss = kl_div_loss * (self.temperature ** 2) # Scale KLDiv loss

            distill_loss = self.alpha * kl_div_loss + (1 - self.alpha) * hard_loss

            self.log(f'{stage_prefix}_kl_div_loss', kl_div_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'{stage_prefix}_hard_loss', hard_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            loss = distill_loss

        # --- Log combined loss ---
        self.log(f'{stage_prefix}_loss', loss, on_step=(stage_prefix=='train'), on_epoch=True, prog_bar=True, sync_dist=True)

        # --- Calculate and log accuracy ---
        metric_obj = None
        if stage_prefix == 'train': metric_obj = self.train_acc
        elif stage_prefix == 'val': metric_obj = self.val_acc
        elif stage_prefix == 'test': metric_obj = self.test_acc

        if metric_obj:
            metric_obj(logits, y)
            self.log(f'{stage_prefix}_acc', metric_obj, on_step=(stage_prefix=='train'), on_epoch=True, prog_bar=True, sync_dist=True)

        # --- Return dictionary (used by test_step) ---
        return {'loss': loss, 'logits': logits, 'targets': y}


    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, stage_prefix='train')['loss']

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, stage_prefix='val')

    def test_step(self, batch, batch_idx):
        results = self._calculate_loss(batch, stage_prefix='test')
        logits = results['logits']
        targets = results['targets']

        preds = torch.argmax(logits, dim=1)

        self.test_step_outputs.append(preds.cpu())
        self.test_step_targets.append(targets.cpu())

        return {'test_loss': results['loss']}

    def on_test_epoch_start(self):
        print("Clearing stored test outputs and targets for new test epoch.")
        self.test_step_outputs.clear()
        self.test_step_targets.clear()

    def on_test_epoch_end(self):
        print("Test epoch end. Processing results for confusion matrix...")

        if not self.test_step_outputs or not self.test_step_targets:
            print("No test outputs collected, skipping confusion matrix.")
            return

        if self.trainer.is_global_zero:
            all_preds = torch.cat(self.test_step_outputs).numpy()
            all_targets = torch.cat(self.test_step_targets).numpy()

            print(f"Collected {len(all_preds)} predictions and targets on rank 0.")

            # Compute confusion matrix
            task = "multiclass" if self.num_classes > 1 else "binary"
            try:
                 cm_tensor = TMF.confusion_matrix(
                     torch.tensor(all_preds), 
                     torch.tensor(all_targets),
                     task=task,
                     num_classes=self.num_classes,
                     normalize='true'
                )
                 cm_tensor = torch.nan_to_num(cm_tensor, nan=0.0)
                 cm_normalized = cm_tensor.numpy() * 100.0
                 print("Normalized Confusion Matrix (Recall %) computed on rank 0.")
            except Exception as e:
                print(f"Error computing confusion matrix on rank 0: {e}")
                self.test_step_outputs.clear()
                self.test_step_targets.clear()
                return


            # Plotting
            plot_dir = self.trainer.default_root_dir
            if self.trainer.logger is not None and hasattr(self.trainer.logger, 'log_dir') and self.trainer.logger.log_dir:
                plot_dir = self.trainer.logger.log_dir

            try:
                 os.makedirs(plot_dir, exist_ok=True)
                 plot_path = os.path.join(plot_dir, "confusion_matrix.png")
                 print(f"Attempting to plot confusion matrix to: {plot_path}")
                 self._plot_confusion_heatmap(cm_normalized, self.class_names, plot_path)
            except Exception as e:
                 print(f"Error during confusion matrix plotting or directory creation: {e}")

        self.test_step_outputs.clear()
        self.test_step_targets.clear()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Hook to modify the checkpoint dictionary before saving.
        During 'distill' stage, we save ONLY the student model's state_dict.
        """
        if self.stage == 'distill' and self.trainer.is_global_zero:
            print("Distillation stage: Modifying checkpoint to save only student model weights.")

            student_state_dict = self.model.state_dict()

            new_state_dict = OrderedDict()
            prefix_to_add = "model."

            for key, value in student_state_dict.items():
                 new_key = prefix_to_add + key
                 new_state_dict[new_key] = value


            # Replace the original state_dict in the checkpoint dictionary
            checkpoint['state_dict'] = new_state_dict
            print(f"Replaced checkpoint state_dict with {len(new_state_dict)} keys from student model (prefixed with 'model.').")

    def _plot_confusion_heatmap(self, cm_normalized, class_names, output_path):
        try:
            num_classes = cm_normalized.shape[0]
            tick_step = 5

            tick_positions = np.arange(0, num_classes, tick_step) + 0.5
            tick_labels = np.arange(0, num_classes, tick_step)

            plt.figure(figsize=(12, 10))

            # Create heatmap
            sns.heatmap(
                cm_normalized,
                annot=False,
                cmap="Blues",
                cbar=True,
                cbar_kws={'label': 'Prediction Percentage (%)'},
                vmin=0, 
                vmax=100
            )

            plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45, ha='right')
            plt.yticks(ticks=tick_positions, labels=tick_labels, rotation=0)

            plt.xlabel("Predicted Class Index")
            plt.ylabel("True Class Index")
            plt.title("Normalized Confusion Matrix (%)")

            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Normalized confusion heatmap saved successfully to {output_path}")

        except Exception as e:
            print(f"Error plotting confusion heatmap: {e}")

    # --- Configure Optimizers ---
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer_name = self.stage_config.get('optimizer', 'AdamW')
        lr = self.stage_config.get('lr', 1e-3) # This is the PEAK LR
        weight_decay = self.stage_config.get('weight_decay', 0.0)

        # Select optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            momentum = self.stage_config.get('momentum', 0.9)
            optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        print(f"Using optimizer: {optimizer_name} with peak_lr={lr}, weight_decay={weight_decay}")

        # --- Configure LR Scheduler ---
        scheduler_config = self.stage_config.get('scheduler')
        if scheduler_config and scheduler_config.get('name') == 'WarmupCosine':
            print("Configuring Warmup + Cosine Annealing LR Scheduler.")

            total_steps = scheduler_config.get('total_steps')
            if total_steps is None:
                 raise ValueError("Scheduler config 'total_steps' is required for WarmupCosine schedule.")

            eta_min = scheduler_config.get('eta_min', 0.0)
            warmup_start_factor = scheduler_config.get('warmup_start_factor', 0.01)

            # --- Determine warmup_steps ---
            warmup_steps = scheduler_config.get('warmup_steps')
            warmup_fraction = scheduler_config.get('warmup_fraction')
            warmup_epochs = scheduler_config.get('warmup_epochs')

            if warmup_steps is not None:
                 print(f"Using explicit warmup_steps from config: {warmup_steps}")
            elif warmup_fraction is not None:
                 if not 0 < warmup_fraction < 1:
                     raise ValueError(f"warmup_fraction ({warmup_fraction}) must be between 0 and 1.")
                 warmup_steps = int(warmup_fraction * total_steps)
                 print(f"Calculated warmup_steps from fraction: {warmup_steps} ({warmup_fraction*100:.1f}% of {total_steps})")
            elif warmup_epochs is not None:
                 try:
                     if self.trainer and self.trainer.datamodule:
                          batches_per_epoch = len(self.trainer.datamodule.train_dataloader())
                          accumulate_grad_batches = self.trainer.accumulate_grad_batches or 1
                          warmup_steps = warmup_epochs * (batches_per_epoch // accumulate_grad_batches)
                          print(f"Calculated warmup_steps from epochs: {warmup_steps} ({warmup_epochs} epochs * {batches_per_epoch // accumulate_grad_batches} steps/epoch)")
                     else:
                          raise ValueError("Trainer/datamodule not available for warmup_steps calculation from epochs.")
                 except Exception as e:
                     raise ValueError(f"Could not estimate batches_per_epoch to calculate warmup_steps from epochs. "
                                      f"Define 'warmup_steps' or 'warmup_fraction' directly. Error: {e}")
            else:
                raise ValueError("Scheduler config requires one of: 'warmup_steps', 'warmup_fraction', or 'warmup_epochs'.")


            if warmup_steps <= 0:
                 print(f"Warning: Calculated warmup_steps ({warmup_steps}) is not positive. Skipping warmup phase.")
                 warmup_steps = 0
            elif warmup_steps >= total_steps:
                raise ValueError(f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps}).")


            # --- Scheduler Instantiation ---
            schedulers_to_sequence = []
            milestones = []

            if warmup_steps > 0:
                 # 1. Linear Warmup Scheduler (only if warmup_steps > 0)
                 scheduler_warmup = LinearLR(
                     optimizer,
                     start_factor=warmup_start_factor,
                     end_factor=1.0,
                     total_iters=warmup_steps
                 )
                 schedulers_to_sequence.append(scheduler_warmup)
                 milestones.append(warmup_steps)

            # 2. Cosine Annealing Scheduler
            cosine_steps = total_steps - warmup_steps
            if cosine_steps <= 0:
                 print(f"Warning: Calculated cosine_steps ({cosine_steps}) is not positive. Cosine Annealing needs duration > 0.")
                 cosine_steps = 1

            scheduler_cosine = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=eta_min
            )
            schedulers_to_sequence.append(scheduler_cosine)


            if len(schedulers_to_sequence) > 1:
                lr_scheduler = SequentialLR(
                    optimizer,
                    schedulers=schedulers_to_sequence,
                    milestones=milestones
                )
                print(f"Using SequentialLR with milestone at step {milestones[0]}.")
            elif len(schedulers_to_sequence) == 1:
                lr_scheduler = schedulers_to_sequence[0]
                print("Using only CosineAnnealingLR (warmup_steps=0).")
            else:
                 print("Warning: No schedulers configured for SequentialLR.")
                 return optimizer


            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }

        elif scheduler_config:
             print(f"Warning: Scheduler '{scheduler_config.get('name')}' configured but not recognized or implemented by 'WarmupCosine' logic. Using no scheduler.")
             return optimizer
        else:
             print("No LR scheduler configured.")
             return optimizer