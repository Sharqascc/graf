import os
import random
import numpy as np

from graf.utils.seeds import set_global_seed

def test_set_global_seed_reproducible():
    set_global_seed(42)
    a = random.randint(0, 1000)
    b = np.random.rand()
    set_global_seed(42)
    assert random.randint(0, 1000) == a
    assert np.random.rand() == b

def test_set_global_seed_sets_pythonhashseed():
    set_global_seed(42)
    assert os.environ["PYTHONHASHSEED"] == "42"

def test_set_global_seed_torch():
    import torch
    set_global_seed(42)
    x = torch.rand(5)
    set_global_seed(42)
    assert torch.allclose(x, torch.rand(5))
