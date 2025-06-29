# Training in 256Hz data and 4s
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import engine_pretraining
import configs
torch.set_float32_matmul_precision("medium")

engine_pretraining.seed_torch(7)


# init model


model = engine_pretraining.LitEEGPT(configs.get_config(**(configs.MODELS_CONFIGS[configs.tag])),
                 USE_LOSS_A =(configs.variant != "A"),
                 USE_LN     =(configs.variant != "B"),
                 USE_SKIP   =(configs.variant != "C"))
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor]

trainer = pl.Trainer(strategy='auto', devices=configs.devices,
                     max_epochs=configs.max_epochs,
                     callbacks=callbacks,
                     log_every_n_steps=configs.config.log_every_n_steps,
                     logger=[pl_loggers.TensorBoardLogger('./logs/', name=f"EEGPT_{configs.tag}_{configs.variant}_tb"),
                             pl_loggers.CSVLogger('./logs/', name=f"EEGPT_{configs.tag}_{configs.variant}_csv")])
trainer.fit(model, configs.train_loader, configs.valid_loader)
