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


model = engine_pretraining.LitEEGPT(configs.get_config(**(configs.MODELS_CONFIGS[configs.tag])),
                 USE_LOSS_A =(configs.variant != "A"),
                 USE_LN     =(configs.variant != "B"),
                 USE_SKIP   =(configs.variant != "C"))

checkpoint_cb = ModelCheckpoint(
    save_top_k=1,                      # save only the best checkpoint
    monitor='val_loss',               # metric to monitor (make sure it's logged)
    mode='min',                       # 'min' if lower is better, 'max' otherwise
    save_last=True,                   # also save the last checkpoint
    dirpath='./checkpoints/',         # directory to save checkpoints
    filename='EEGPT_{epoch:03d}-{val_loss:.4f}'  # naming pattern
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

