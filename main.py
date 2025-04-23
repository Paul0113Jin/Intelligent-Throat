# main.py
import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
from datamodule import MyDataModule
from modelmodule import ModelModule

torch.set_float32_matmul_precision('high')

def main():
    # --- 1. Load Configuration ---
    config_path = 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return

    stage = config.get('stage')
    if stage not in ['pretrain', 'finetune', 'distill', 'test']:
        print(f"Error: Invalid stage '{stage}' in config.yaml. Must be one of: pretrain, finetune, distill, test.")
        return

    print(f"--- Running Stage: {stage.upper()} ---")

    # --- Seed for reproducibility ---
    pl.seed_everything(config.get('seed', 42), workers=True)

    # --- 2. Initialize DataModule ---
    data_module = MyDataModule(config=config, stage=stage)

    # --- 3. Initialize ModelModule ---
    if stage == 'finetune':
        pretrained_ckpt_path = config['finetune'].get('pretrained_ckpt_path')
        if not pretrained_ckpt_path or not os.path.exists(pretrained_ckpt_path):
            raise FileNotFoundError(f"Finetuning stage requires 'pretrained_ckpt_path' in config['finetune'] pointing to a valid file. Path given: {pretrained_ckpt_path}")

        print(f"Loading ModelModule from checkpoint for finetuning: {pretrained_ckpt_path}")
        model_module = ModelModule.load_from_checkpoint(
            checkpoint_path=pretrained_ckpt_path,
            model_config=config['model'],    
            stage_config=config['finetune'], 
            stage='finetune',                
        )
        print("ModelModule loaded and configured for finetuning.")
    else:
        model_module = ModelModule(
            model_config=config['model'],
            stage_config=config[stage],
            stage=stage
        )
        print(f"ModelModule instantiated directly for stage: {stage}")

    # --- 4. Configure Callbacks ---
    callbacks = []
    output_dir = config[stage].get('output_dir', f'lightning_logs/{stage}')
    os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists

    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    # Add Learning Rate Monitor if a scheduler is used
    if config[stage].get('scheduler'):
        callbacks.append(LearningRateMonitor(logging_interval='step'))


    # Model Checkpointing (only for training stages)
    if stage in ['pretrain', 'finetune', 'distill']:
        monitor_metric = config[stage].get('monitor_metric', 'val_acc')
        monitor_mode = config[stage].get('monitor_mode', 'max')
        print(f"Configuring ModelCheckpoint to monitor '{monitor_metric}' in '{monitor_mode}' mode.")

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename=config[stage].get('checkpoint_filename', f'{stage}-{{epoch:02d}}-{{{monitor_metric}:.4f}}'),
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=config[stage].get('save_top_k', 1),
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)

        early_stopping_patience = config[stage].get('early_stopping_patience')
        if early_stopping_patience and early_stopping_patience > 0:
            print(f"Configuring EarlyStopping to monitor '{monitor_metric}' in '{monitor_mode}' mode with patience {early_stopping_patience}.")
            early_stop_callback = EarlyStopping(
                monitor=monitor_metric,                 
                patience=early_stopping_patience,       
                mode=monitor_mode,                      
                verbose=True,
                min_delta=0.001
            )
            callbacks.append(early_stop_callback)
        else:
            print("EarlyStopping not configured or patience <= 0 for this stage.")

    # --- 5. Initialize Trainer ---
    trainer_config = config.get('trainer', {})
    if stage in ['pretrain', 'finetune', 'distill'] and 'max_epochs' in config[stage]:
        trainer_config['max_epochs'] = config[stage]['max_epochs']
        print(f"Overriding trainer max_epochs with stage setting: {trainer_config['max_epochs']}")
    elif stage == 'test':
         trainer_config.pop('max_epochs', None)


    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **trainer_config
    )

    # --- 6. Run the appropriate stage ---
    if stage in ['pretrain', 'distill']:
        print(f"Starting {stage} training...")
        trainer.fit(model=model_module, datamodule=data_module)

    elif stage == 'finetune':
        print(f"Starting finetuning training...")
        trainer.fit(model=model_module, datamodule=data_module)

    elif stage == 'test':
        test_ckpt_path = config['test'].get('ckpt_path')
        if test_ckpt_path and os.path.exists(test_ckpt_path):
            print(f"Starting testing with checkpoint: {test_ckpt_path}")
            test_results = trainer.test(model=model_module, datamodule=data_module, ckpt_path=test_ckpt_path)
            print("--- Test Results ---")
            print(test_results)
            results_path = os.path.join(output_dir, 'test_results.yaml')
            with open(results_path, 'w') as f:
                 yaml.dump(test_results, f)
            print(f"Test results saved to: {results_path}")

        elif test_ckpt_path == 'best' and stage in ['pretrain', 'finetune', 'distill']:
             print("Warning: Testing with ckpt_path='best' requires knowing the best path from the relevant training stage. Please specify the exact path in config['test']['ckpt_path'].")

        else:
            print(f"Error: Checkpoint path '{test_ckpt_path}' for testing not found or specified in config.")


if __name__ == '__main__':
    main()