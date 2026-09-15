####################################################################################################
#                                          dataModules.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Lightning data modules wrapping the simulator: on-the-fly training batches and a        #
#          validation stream redrawn every pass (same mechanism as training); the val simulator    #
#          is seeded once and persists, so pass k yields identical data across all runs.           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, IterableDataset

from cnmmusic.data.simulator import NarrowbandSimulator


#**************************************************************************************************#
#                                      Class SimulatorStream                                       #
#**************************************************************************************************#
#                                                                                                  #
# Iterable dataset yielding ready-made batch dicts from the simulator (batching happens inside     #
# the simulator, so the DataLoader runs with batch_size=None).                                     #
#                                                                                                  #
#**************************************************************************************************#
class SimulatorStream(IterableDataset):
    def __init__(self, sim_config, batch, steps, sim=None):
        super().__init__()
        self.sim_config = sim_config
        self.batch = batch
        self.steps = steps
        self.sim = sim              # persistent simulator: fresh draws on every re-iteration

    def __iter__(self):
        sim = self.sim if self.sim is not None else NarrowbandSimulator(self.sim_config)
        if self.steps is None or self.steps < 0:               # infinite stream (steps-based runs)
            while True:
                yield sim.sample(self.batch)
        else:
            for _ in range(self.steps):
                yield sim.sample(self.batch)


#**************************************************************************************************#
#                                       Class DOADataModule                                        #
#**************************************************************************************************#
class DOADataModule(pl.LightningDataModule):
    def __init__(self, sim_config=None, batch=32, steps_per_epoch=200, val_batches=16,
                 val_seed=1234):
        super().__init__()
        self.sim_config = dict(sim_config or {})
        self.batch = batch
        self.steps_per_epoch = steps_per_epoch
        self.val_batches = val_batches
        self.val_seed = val_seed
        self._val = None
        self._train = None

    def setup(self, stage=None):
        if self._val is None:       # ONE seeded simulator, drawn from anew every val pass
            self._val = NarrowbandSimulator(dict(self.sim_config, seed=self.val_seed))
        if self._train is None:     # persistent train simulator: its generator state is
            self._train = NarrowbandSimulator(self.sim_config)   # checkpointed (see below)

    def train_dataloader(self):
        return DataLoader(SimulatorStream(self.sim_config, self.batch, self.steps_per_epoch,
                                          sim=self._train), batch_size=None)

    #*********************************************************************#
    #   checkpointed stream state (perfect data continuation on resume)   #
    #*********************************************************************#
    def state_dict(self):
        """Both simulators' generator states -> the Lightning checkpoint: a resumed run
        CONTINUES the train stream (no replay from the seed) and keeps the val pass sequence
        aligned. Requires num_workers = 0 (in-process simulators; the project default)."""
        self.setup()
        return {'train_rng': self._train.rng.get_state(), 'val_rng': self._val.rng.get_state()}

    def load_state_dict(self, state):
        self.setup()
        self._train.rng.set_state(state['train_rng'])
        self._val.rng.set_state(state['val_rng'])

    def val_dataloader(self):
        return DataLoader(SimulatorStream(self.sim_config, self.batch, self.val_batches,
                                          sim=self._val), batch_size=None)
