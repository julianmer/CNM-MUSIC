####################################################################################################
#                                             sweep.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Weights & Biases hyperparameter sweeps over the training pipeline (auth via the         #
#          WANDB_API_KEY environment variable; never hardcode keys).                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import pytorch_lightning as pl
import wandb

from cnmmusic.train import Pipeline


#***********#
#   sweep   #
#***********#
sweep_config = {
    'method': 'random',
    'name': 'cnm_sweep',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 3e-3},
        'z_dim': {'values': [4, 16, 32]},
        'online': {'values': [True]},
    },
}


#*********#
#   run   #
#*********#
if __name__ == '__main__':
    pl.seed_everything(42)
    wandb.login()                                   # reads WANDB_API_KEY from the environment
    sweep_id = wandb.sweep(sweep_config, project='cnmmusic')
    wandb.agent(sweep_id, Pipeline().main)
