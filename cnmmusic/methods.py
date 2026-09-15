####################################################################################################
#                                            methods.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: The single method registry: every estimator (classical, oracle, bound, learned) is      #
#          built from a spec string. Classical: 'music', 'root-music', ... Learned methods take    #
#          their checkpoint after a colon: 'cnm:checkpoints/x.ckpt'. Display names match           #
#          the fixed plot styles in visualize.METHOD_STYLES.                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from cnmmusic.estimators.classical import (MUSIC, RootMUSIC, ESPRIT, MVDR, MLE, FreqMUSIC,
                                           FreqMVDR, FreqCascade, FreqCascadeMVDR, FreqESPRIT,
                                           SpatialSmoothingMUSIC, OracleMUSIC,
                                           OracleRootMUSIC)
from cnmmusic.estimators.nearfield import (NearFieldMUSIC, NearFieldMVDR, NearFieldCascade,
                                          NearFieldCascadeMVDR, NearFieldMLE)
from cnmmusic.estimators.neural import (CNMEstimator, SubspaceNetEstimator, DAMUSICEstimator,
                                        GridCNNEstimator, CNMDAMUSICEstimator)

CLASSICAL = {
    'mvdr': ('MVDR', MVDR),
    'music': ('MUSIC', MUSIC),
    'root-music': ('Root-MUSIC', RootMUSIC),
    'esprit': ('ESPRIT', ESPRIT),
    'ss-music': ('SS-MUSIC', SpatialSmoothingMUSIC),
    'mle': ('MLE', MLE),
    # the same joint 2-D MUSIC scan, second axis per scenario: (theta, f) or (theta, r)
    'f-music': ('MUSIC (2D)', FreqMUSIC),
    '2d-music': ('MUSIC (2D)', NearFieldMUSIC),
    'cascade': ('MUSIC (Cascade)', NearFieldCascade),
    # the same joint scans under the Capon back-end
    'f-mvdr': ('MVDR (2D)', FreqMVDR),
    'f-cascade': ('MUSIC (Cascade)', FreqCascade),
    'f-cascade-mvdr': ('MVDR (Cascade)', FreqCascadeMVDR),
    # closed-form joint angle-frequency estimation: space-time ESPRIT, no grid
    'f-esprit': ('ESPRIT (JAFE)', FreqESPRIT),
    '2d-mvdr': ('MVDR (2D)', NearFieldMVDR),
    'cascade-mvdr': ('MVDR (Cascade)', NearFieldCascadeMVDR),
    'nf-mle': ('NF-MLE', NearFieldMLE),
    'oracle': ('Oracle', OracleMUSIC),
    'oracle-root': ('Oracle-Root', OracleRootMUSIC),
}

LEARNED = {
    # one checkpoint, the native corrected-null scan; further back-ends (MVDR, ...)
    # get their own specs here
    'cnm': ('CNM-MUSIC', lambda ck, g, c: CNMEstimator(ck, g, c)),
    'cnm-music': ('CNM-MUSIC', lambda ck, g, c: CNMEstimator(ck, g, c, method='music')),
    # uncertainty-gated correction: needs calibrate() on base-point scenes before use
    'cnm-gated': ('CNM (gated)', lambda ck, g, c: CNMEstimator(ck, g, c, gate=True)),
    # the corrected manifold under another back-end: Capon instead of the null spectrum
    'cnm-mvdr': ('CNM-MVDR', lambda ck, g, c: CNMEstimator(ck, g, c, method='mvdr')),
    # the corrected manifold inside DA-MUSIC's back-end (two checkpoints, '+'-joined)
    'cnm-damusic': ('CNM-DA-MUSIC', CNMDAMUSICEstimator),
    # the corrected manifold on SubspaceNet's learned subspace ('<cnm>+<subspacenet>')
    'cnm-ssn': ('CNM-SubspaceNet-MUSIC',
                lambda ck, g, c: CNMEstimator(*ck.split('+')[:1], g, c,
                                              subspace=ck.split('+')[1])),
    'cnm-ssn-mvdr': ('CNM-SubspaceNet-MVDR',
                     lambda ck, g, c: CNMEstimator(*ck.split('+')[:1], g, c, method='mvdr',
                                                   subspace=ck.split('+')[1])),
    'subspacenet': ('SubspaceNet', SubspaceNetEstimator),
    'subspacenet-music': ('SubspaceNet-MUSIC',
                          lambda ck, g, c: SubspaceNetEstimator(ck, g, c, method='music')),
    'damusic': ('DA-MUSIC', DAMUSICEstimator),
    'gridcnn': ('GridCNN', GridCNNEstimator),
}


#***********#
#   build   #
#***********#
def build(spec, geom, config):
    """
    'music' | 'crb' | 'cnm:checkpoints/x.ckpt' -> (display_name, estimator_or_None,
    checkpoint_or_None). 'crb' returns estimator None (the harness computes the bound itself).
    """
    name, _, ckpt = spec.partition(':')
    name = name.strip().lower()
    if name == 'crb':
        return 'CRB', None, None
    if name == 'zzb':
        return 'ZZB', None, None
    if name in CLASSICAL:
        display, cls = CLASSICAL[name]
        return display, cls(geom, config=config), None
    if name in LEARNED:
        if not ckpt:
            raise ValueError(f"learned method '{name}' needs a checkpoint: '{name}:path.ckpt'")
        display, factory = LEARNED[name]
        return display, factory(ckpt, geom, config), ckpt
    raise ValueError(f'unknown method: {name} (known: '
                     f'{sorted(CLASSICAL) + sorted(LEARNED) + ["crb"]})')
