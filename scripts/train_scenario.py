####################################################################################################
#                                        train_scenario.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Train a model on one scenario of the ladder (cnmmusic/scenarios.py).                    #
#          Run: python scripts/train_scenario.py --scenario s3_mismatch [--model cnm] [--online]   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import sys

from cnmmusic.scenarios import SCENARIOS
from cnmmusic.train import Pipeline


#*********#
#   run   #
#*********#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', required=True, choices=sorted(SCENARIOS))
    parser.add_argument('--geometry', default=None,             # orthogonal to the scenario:
                        choices=['ula', 'uca', 'ura', 'nula', 'nested', 'random_planar'])   # overrides
    parser.add_argument('--fov_deg', type=float, default=None)  # the dict's geometry; FOV
    parser.add_argument('--grid_size', type=int, default=None)  # shared scan grid override
    parser.add_argument('--model', default='cnm')               # follows via fov_deg (map or
                                                                # this scalar override)
    parser.add_argument('--backend', default=None, choices=['none', 'mlp', 'sphere'])
    parser.add_argument('--encoder', default=None,
                        choices=['snapshot', 'covariance', 'lagcov', 'snapattn', 'meanset'])
    parser.add_argument('--train_cov', default=None, choices=['empirical', 'true'])
    parser.add_argument('--train_En', default=None, choices=['empirical', 'true'])
    parser.add_argument('--z_dim', type=int, default=None)
    parser.add_argument('--n_hard', type=int, default=None)
    parser.add_argument('--n_harm', type=int, default=None)
    parser.add_argument('--excl', type=float, default=None)     # pin the ring (None = auto)
    parser.add_argument('--load_model', default=None)           # checkpoint path to resume from
    parser.add_argument('--resume', default=None)               # wandb run id: proper resume
                                                                # (same row, stream continuation)
    parser.add_argument('--loss', default=None, choices=['rank', 'nll', 'nlls', 'ce', 'ce-beam',
                                                         'mspe', 'rmspe'])   # last two: baselines
    parser.add_argument('--tau_nll', type=float, default=None)
    parser.add_argument('--corr_mode', default=None, choices=['mult', 'add', 'free'])
    parser.add_argument('--corr_cond', default=None, choices=['concat', 'film', 'attn'])
    parser.add_argument('--coord_enc', default=None, choices=['harm', 'lff', 'raw'])
    parser.add_argument('--pin_aux', action='store_true')   # aux axes at nominal in the
    parser.add_argument('--eps_scale', type=float, default=None)   # target-bump width scale
                                                            # loss (theta-only supervision)
    parser.add_argument('--val_estimators', default=None)      # comma-separated, e.g. "music"
    parser.add_argument('--range', type=float, nargs=2, default=None,   # override the scenario
                        metavar=('LO', 'HI'))                           # spans (half-wavelengths
    parser.add_argument('--freq', type=float, nargs=2, default=None,    # / f/fc units)
                        metavar=('LO', 'HI'))
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--callback', default='val/music/doa_rmspe_deg')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)   # baselines' Adam
    parser.add_argument('--batch', type=int, default=None)            # per-step batch
    parser.add_argument('--trueBatch', type=int, default=None)        # effective batch,
                                                                      # reached by
                                                                      # gradient accumulation
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    sys.argv = sys.argv[:1]                 # the Pipeline parser must not re-read our flags

    cfg = {**SCENARIOS[args.scenario], 'model': args.model, 'scenario': args.scenario,
           'max_steps': args.max_steps, 'online': args.online, 'callback': args.callback,
           'run_tag': args.tag}
    if args.geometry is not None:
        cfg['geometries'] = [args.geometry]
        if args.geometry == 'random_planar':        # per-batch redraws would break training
            cfg['freeze_geometry'] = True
    if args.fov_deg is not None:
        cfg['fov_deg'] = args.fov_deg
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.range is not None:
        cfg['range'] = tuple(args.range)
    if args.freq is not None:
        cfg['freq'] = tuple(args.freq)
    if args.pin_aux:
        cfg['pin_aux'] = True
    if args.val_estimators is not None:
        cfg['val_estimators'] = args.val_estimators.split(',')
    for key in ('backend', 'encoder', 'train_cov', 'train_En', 'z_dim', 'n_hard', 'n_harm',
                'excl', 'load_model', 'resume', 'loss', 'tau_nll', 'corr_mode',
                'corr_cond', 'coord_enc', 'grid_size', 'weight_decay', 'eps_scale',
                'batch', 'trueBatch'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.model == 'subspacenet':         # tau-lag autocorrelation needs T > tau
        lo, hi = cfg['snapshots']
        cfg['snapshots'] = (max(lo, 10), hi)
    Pipeline().main(cfg)
