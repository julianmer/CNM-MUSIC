####################################################################################################
#                                           framework.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: The abstract Lightning framework base and the model factory.                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import pytorch_lightning as pl
import torch

from abc import ABC, abstractmethod


#*************************#
#   batch (t, d) groups   #
#*************************#
def condition_groups(batch):
    """
    Yields (idx, T, d) for every unique (snapshots, n_src) pair in the batch. Simulator batches
    mix T and d per element (padded); slicing each group back to its true sizes gives dense
    sub-batches no model ever sees padding in.
    """
    B = batch['X'].shape[0]
    Ts = torch.as_tensor(batch['snapshots']).reshape(-1)
    ds = torch.as_tensor(batch['n_src']).reshape(-1)
    Ts = Ts.expand(B) if Ts.numel() == 1 else Ts
    ds = ds.expand(B) if ds.numel() == 1 else ds
    pairs = torch.stack([Ts, ds], dim=1)
    uniq, inv = torch.unique(pairs, dim=0, return_inverse=True)
    for g in range(uniq.shape[0]):
        idx = (inv == g).nonzero(as_tuple=True)[0]
        yield idx, int(uniq[g, 0]), int(uniq[g, 1])


def sub_batch(batch, idx, T, d):
    """The dense sub-batch of one (T, d) group: rows idx, snapshots cut to T, sources to d."""
    B = batch['X'].shape[0]
    out = {}
    for key, v in batch.items():
        out[key] = v[idx] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == B \
            and key != 'grid' else v
    out['X'] = out['X'][..., :T]
    out['doas'] = out['doas'][..., :d]
    out['ranges'] = out['ranges'][..., :d]
    out['a_true_doa'] = out['a_true_doa'][..., :d]
    out['R_s'] = out['R_s'][..., :d, :d]
    if torch.is_tensor(out.get('freqs')) and out['freqs'].ndim == 2:   # per-source carriers
        out['freqs'] = out['freqs'][..., :d]
    out['n_src'] = d
    out['snapshots'] = T
    return out


#**************************************************************************************************#
#                                         Class Framework                                          #
#**************************************************************************************************#
class Framework(pl.LightningModule, ABC):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters(config)

    @abstractmethod
    def forward(self, batch):
        ...

    #**************************************************************************#
    #   global rng states in the checkpoint (perfect continuation on resume)   #
    #**************************************************************************#
    def on_save_checkpoint(self, checkpoint):
        checkpoint['rng'] = {'torch': torch.get_rng_state(),
                             'cuda': (torch.cuda.get_rng_state_all()
                                      if torch.cuda.is_available() else [])}

    def on_load_checkpoint(self, checkpoint):
        rng = checkpoint.get('rng')          # absent in pre-feature checkpoints -> fresh RNG
        if rng is None:
            return
        torch.set_rng_state(rng['torch'].cpu())
        if torch.cuda.is_available() and len(rng['cuda']) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])


#*******************#
#   get_framework   #
#*******************#
def get_framework(model_name, **config):
    if model_name in ('cnm', 'cnmmusic'):
        from cnmmusic.models.frameworkCNM import CNMFramework
        return CNMFramework(**config)
    if model_name in ('subspacenet', 'damusic', 'gridcnn'):
        from cnmmusic.models.frameworkBaselines import BaselinesFramework
        return BaselinesFramework(baseline=model_name, **config)
    raise ValueError(f'unknown model: {model_name}')
