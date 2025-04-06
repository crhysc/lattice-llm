# resolvers.py

"""
Hydra Resolver Registry
-----------------------

This module defines and registers all custom resolvers used in the configuration system.
Resolvers allow dynamic computation or transformation of config values using expressions
like `${resolver_name:args}` inside YAML config files.

Usage:
    Import and call `register_resolvers()` at the top of your main script (before @hydra.main):
    
        from resolvers import register_resolvers
        register_resolvers()

    Then you can use expressions like `${torch_dtype:float16}` or `${data_split:80:20}`
    in your YAML config.

Resolvers Defined:
    - torch_dtype: Maps a string like "float16" to the corresponding torch.dtype object.
    - data_split: Converts a ratio string like "80:20" into normalized train/test fractions.
"""

from omegaconf import OmegaConf
from model_utils import torch_dtype_resolver
from data_utils import data_split_resolver


def register_resolvers():
    """
    Registers all custom resolvers with OmegaConf so they can be used in Hydra configs.
    This function should be called before Hydra parses any config (i.e., before @hydra.main).
    """
    OmegaConf.register_new_resolver("torch_dtype", torch_dtype_resolver)
