import numpy as np

from epsilon_transformers.process.processes import ZeroOneR
from epsilon_transformers.process.PartitionLattice import coarse_grain_process

def test_coarse_grain_processs():
    z1r = ZeroOneR()
    expected = np.array([[[.25, 0.5],
                            [0.0, 0.0]],
                            
                            [[.25, 0.0],
                            [1.0,0.0]]])
    actual = coarse_grain_process(hmm_tensor=z1r.transition_matrix, state_groups=[[0,2],[1]]).transition_matrix
    assert np.allclose(expected, actual)
    