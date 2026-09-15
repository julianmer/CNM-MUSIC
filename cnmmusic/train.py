####################################################################################################
#                                             train.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Train and evaluate the neural models implemented in Pytorch Lightning.                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import math
import os

import pytorch_lightning as pl
import torch
import wandb

LOG_DIR = './logs'                          # everything (wandb, checkpoints) lives here

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from cnmmusic.config import DEFAULT_CONFIG, parse_cli, sim_config
from cnmmusic.data.dataModules import DOADataModule
from cnmmusic.models.framework import get_framework


#**************************************************************************************************#
#                                          Class Pipeline                                          #
#**************************************************************************************************#
class Pipeline:
    def __init__(self):
        pl.seed_everything(DEFAULT_CONFIG['seed'])
        self.default_config = DEFAULT_CONFIG

    #**********#
    #   main   #
    #**********#
    def main(self, config_updates=None):
        merged = parse_cli(self.default_config)
        merged.update(config_updates or {})

        run_name = '-'.join(filter(None, (merged.get('model'), merged.get('scenario'),
                                          merged.get('run_tag'))))
        os.makedirs(LOG_DIR, exist_ok=True)
        resume_id = merged.get('resume') or None     # proper resume: SAME wandb row (logging
        wandb.init(config=merged, project='cnmmusic', entity=merged.get('entity'),   # appends),
                   name=run_name or None, dir=LOG_DIR,                # checkpoint auto-located
                   id=resume_id, resume='must' if resume_id else None,
                   mode='online' if merged['online'] else 'offline')
        config = dict(wandb.config)
        if resume_id and not config['load_model']:
            config['load_model'] = os.path.join(LOG_DIR, 'lightning_logs', resume_id,
                                                'checkpoints', 'last.ckpt')

        if not isinstance(config['fov_deg'], dict):     # scalar override: one +-half-FOV for
            config['fov_deg'] = {g: float(config['fov_deg'])       # whatever geometry runs
                                 for g in config['geometries']}
        fov = math.radians(config['fov_deg'][config['geometries'][0]]) - 1e-3
        # scan past the FOV (scan_margin_deg) with the grid step preserved: truths stay
        # inside the FOV, but edge nulls are whole and edge peaks are interior maxima
        scan = fov + math.radians(config['scan_margin_deg'])
        n_scan = int(round(config['grid_size'] * scan / fov))
        model = get_framework(config['model'],                       # coarray scenes are
                              M=config.get('M_virtual') or config['M'],  # virtual-M sized
                              z_dim=config['z_dim'],
                              n_src=max(config['n_src']), snapshots=max(config['snapshots']),
                              hidden=config['hidden'], encoder=config['encoder'],
                              grid_size=n_scan, theta_range=(-scan, scan),
                              lr=config['lr'], optimizer=config['optimizer'],
                              weight_decay=config['weight_decay'],
                              range_span=config['range'],
                              freq_span=tuple(config['freq']), backend=config['backend'],
                              train_cov=config['train_cov'], train_En=config['train_En'],
                              loss=config['loss'], tau_nll=config['tau_nll'],
                              corr_mode=config['corr_mode'],
                              corr_cond=config['corr_cond'], coord_enc=config['coord_enc'],
                              n_neg=config['n_neg'],
                              margin=config['margin'], excl=config['excl'], tau=config['tau'],
                              n_probe=config['n_probe'], n_hard=config['n_hard'],
                              n_harm=config['n_harm'],
                              min_sep_rad=math.radians(config['min_sep_deg']),
                              pin_aux=config.get('pin_aux', False),
                              eps_scale=config.get('eps_scale', 1.0),
                              val_estimators=tuple(config['val_estimators']))
        data = DOADataModule(sim_config(config), batch=config['batch'], steps_per_epoch=-1)

        wandb_logger = WandbLogger(save_dir=LOG_DIR, offline=not config['online'])
        checkpoint_cb = ModelCheckpoint(monitor=config['callback'], save_last=True,
                                        every_n_train_steps=config['val_every_steps'])
        lr_monitor = LearningRateMonitor(logging_interval='step')

        torch.set_float32_matmul_precision('medium')
        accel = ({'accelerator': 'gpu', 'devices': [config['gpu_selection']]}
                 if torch.cuda.is_available() else {'accelerator': 'cpu'})   # MPS lacks complex
        trainer = pl.Trainer(max_steps=config['max_steps'] if config['max_steps'] > 0 else -1,
                             max_epochs=-1,
                             default_root_dir=LOG_DIR,
                             val_check_interval=config['val_every_steps'],
                             check_val_every_n_epoch=None,
                             logger=wandb_logger,
                             callbacks=[checkpoint_cb, lr_monitor],
                             log_every_n_steps=10,
                             accumulate_grad_batches=max(config['trueBatch'] // config['batch'], 1),
                             **accel)
        trainer.fit(model, data, ckpt_path=config['load_model'] or None)
        return checkpoint_cb.best_model_path


#*********#
#   run   #
#*********#
if __name__ == '__main__':
    Pipeline().main()
