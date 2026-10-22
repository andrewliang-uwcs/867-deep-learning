# -*- coding: utf-8 -*-
"""Learnable component of the model-based ISAC transceiver."""
import torch.nn as nn
class PertNet(nn.Module):
    """Holds the learnable inter-antenna spacing vector d (shape (K,1), metres).

    There are no layers: d is used directly by the precoder and the sensing receiver in
    lib/functions.py and lib/scene_pool.py, and is learned by backpropagating through them.
    """
    def __init__(self,init_vector):
        """init_vector: initial spacing, e.g. lambda/2 * ones((K,1)). Wrapped (not copied) in
        an nn.Parameter, so on CPU it shares memory with the tensor passed in."""
        super(PertNet, self).__init__()
        #Define vector to optimize
        self.d = nn.Parameter(init_vector)