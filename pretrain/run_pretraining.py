# Training in 256Hz data and 4s
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import engine_pretraining
import configs
torch.set_float32_matmul_precision("medium")

engine_pretraining.seed_torch(7)


# init model


model = engine_pretraining.LitEEGPT(
                models_configs=configs.get_config(**(configs.MODELS_CONFIGS[configs.tag])),
                positive_valid_loader=configs.positive_valid_loader,
                negative_valid_loader=configs.negative_valid_loader,
                positive_test_loader=configs.positive_test_loader,
                negative_test_loader=configs.negative_test_loader,
                USE_LOSS_A =(configs.variant != "A"),
                USE_LN     =(configs.variant != "B"),
                USE_SKIP   =(configs.variant != "C"))

checkpoint_cb = ModelCheckpoint(
    save_top_k=5,                      # save only the best checkpoint
    monitor='val/F1',               # metric to monitor (make sure it's logged)
    mode='max',                       # 'min' if lower is better, 'max' otherwise
    save_last=True,                   # also save the last checkpoint
    dirpath='./checkpoints/',         # directory to save checkpoints
    filename='EEGPT_{epoch:03d}_{val_F1:.4f}'  # naming pattern
)

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor, checkpoint_cb]

trainer = pl.Trainer(strategy='auto', devices=configs.devices,
                     max_epochs=configs.max_epochs,
                     callbacks=callbacks,
                     log_every_n_steps=configs.steps_per_epoch,
                     logger=[pl_loggers.TensorBoardLogger('./logs/', name=f"EEGPT_{configs.tag}_{configs.variant}_tb"),
                             pl_loggers.CSVLogger('./logs/', name=f"EEGPT_{configs.tag}_{configs.variant}_csv")])
trainer.fit(model, configs.train_loader, configs.valid_loader)

# Get best checkpoint path
print("Best model checkpoint path:", checkpoint_cb.best_model_path)

# Optional: get last checkpoint
print("Last checkpoint path:", checkpoint_cb.last_model_path)

