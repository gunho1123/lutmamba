import torch
import torch.nn as nn


DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    """Find all layers of specified types in a module, including deeply nested ones.
    
    Args:
        module: The module to search in
        layers: List of layer types to search for
        name: Current name prefix for the search
    
    Returns:
        Dictionary mapping layer names to layer modules
    """
    res = {}
    
    # Use named_modules() to get all modules including deeply nested ones
    for name1, child in module.named_modules():
        if type(child) in layers:
            # Use the full path from named_modules
            full_name = name + '.' + name1 if name != '' else name1
            res[full_name] = child
    
    return res
