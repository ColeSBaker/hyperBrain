import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules.loss


def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])


def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
                [
                    d
                    for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d)) and d.isnumeric()
                    ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser



def checkpoint_params(model):
    """
    creates a saved checkpoint of model params that can be used to see if a model has changed
    """

    # get a list of params that are allowed to change
    params = [ np for np in model.named_parameters() if np[1].requires_grad ]
    #take a copy
    initial_params = [ (name, p.clone()) for (name, p) in params ]
    return initial_params

def has_model_changed(model,initial_params):
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    new_params=checkpoint_params(model)
    # print(initial_params,'initial_params')
    # print(new_params,'initial_params')
    # check if variables have changed
    all_ch=True
    any_ch=False
    for (_, p0), (name, p1) in zip(initial_params, new_params):
        eq= torch.equal(p0.to(device), p1.to(device))
        # print(eq,'IS EQUAL')
        if eq:
            all_ch=False
        else:
            any_ch=True
    if all_ch:
        assert any_ch

    # print(all_ch,any_ch)

    return all_ch,any_ch