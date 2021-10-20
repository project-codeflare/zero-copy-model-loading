#
#  Copyright (c) 2021 IBM Corp.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
PyTorch model rewrites related to zero-copy model loading. These rewrites allow users to
separate a model into its weights and graph, so that the weights can be loaded via a
zero-copy mechanism such as `ray.get()`, then plugged into an empty version of the graph.
"""

import copy
import torch
from typing import Dict, List, Tuple


def extract_tensors(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays, and return the stripped model and tensors.

    :param m: Root node of a PyTorch model encoded as a graph of subclasses of
        :class:`torch.nn.Module`
    :type m: torch.nn.Module

    :returns: A tuple with two elements:

        * A deep copy of `m` in which all weight tensors have been replaced by `None`
        * The tensors that were removed from the copy of `m`, encoded as a list of
          dictionaries. Each dictionary holds the tensors associated with a single
          :class:`torch.nn.Module` in the model's graph, indexed by parameter name.
          The dictionaries occur in the order returned by :func:`m.named_modules`
    """
    tensors = []
    for _, module in m.named_modules():
        # Store the tensors in Python dictionaries
        params = {
            name: torch.clone(param).detach().numpy()
            for name, param in module.named_parameters(recurse=False)
        }
        buffers = {
            name: torch.clone(buf).detach().numpy()
            for name, buf in module.named_buffers(recurse=False)
        }
        tensors.append({"params": params, "buffers": buffers})

    # Make a copy of the original model and strip all tensors and
    # temporary buffers out of the copy.
    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in (
                [name for name, _ in module.named_parameters(recurse=False)]
                + [name for name, _ in module.named_buffers(recurse=False)]):
            setattr(module, name, None)

    # Make sure the copy is configured for inference.
    m_copy.train(False)
    return m_copy, tensors


def replace_tensors(m: torch.nn.Module, tensors: List[Dict]):
    """
    The inverse operation of :func:`extract_tensors`. Restores the tensors that
    :func:`extract_tensors` stripped out of a  PyTorch model. This restore operation
    involves zero copying of data and results in a model that can be immediately
    used for CPU-based inference. To avoid copying, this function modifies the target
    model in place.

    :param m: Root node of a PyTorch model encoded as a graph of subclasses of
        :class:`torch.nn.Module`. Usually this parameter contains a model that has been
        stripped of its weights by :funct:`extract_tensors`. **Modified in place.**
        If any weights are present in `m`, this function will replace them.
    :param tensors: The tensors to be inserted into `m`, encoded as a list of
        dictionaries. Each dictionary holds the tensors associated with a single
        :class:`torch.nn.Module` in the model's graph, indexed by parameter name.
        The dictionaries occur in the order returned by :func:`m.named_modules`
    """
    with torch.inference_mode():
        modules = [module for _, module in m.named_modules()]
        for module, tensor_dict in zip(modules, tensors):
            # There are separate APIs to set parameters and buffers.
            for name, array in tensor_dict["params"].items():
                module.register_parameter(
                    name, torch.nn.Parameter(torch.as_tensor(array)))
            for name, array in tensor_dict["buffers"].items():
                module.register_buffer(name, torch.as_tensor(array))


def replace_tensors_direct(m: torch.nn.Module, tensors: List[Dict]):
    """
    A version of :func:`replace_tensors` that takes a faster but slightly dangerous
    shortcut.

    Like :func:`replace_tenosrs`, this function restores the tensors that
    :func:`extract_tensors` stripped out of a PyTorch model. However, this function
    skips the step of wrapping the restored tensors in ``torch.nn.Parameters`` objects.
    Skipping this step makes the restore operation go about 20% faster in testing on
    ``bert-base-uncased``, but **may impact the correctness of some models**.
    Be sure to test this method carefully before using it on a particular PyTorch model.

    Like :func:`replace_tensors`, this function modifies the model in place to avoid
    copying data.

    :param m: Root node of a PyTorch model encoded as a graph of subclasses of
        :class:`torch.nn.Module`. Usually this parameter contains a model that has been
        stripped of its weights by :funct:`extract_tensors`. **Modified in place.**
        If any weights are present in `m`, this function will replace them.
    :param tensors: The tensors to be inserted into `m`, encoded as a list of
        dictionaries. Each dictionary holds the tensors associated with a single
        :class:`torch.nn.Module` in the model's graph, indexed by parameter name.
        The dictionaries occur in the order returned by :func:`m.named_modules`
    """
    with torch.inference_mode():
        modules = [module for _, module in m.named_modules()]
        for module, tensor_dict in zip(modules, tensors):
            # There are separate APIs to set parameters and buffers.
            for name, array in tensor_dict["params"].items():
                # Super fast, somewhat risky version avoids
                # wrapping parameters in Parameters objects.
                module._parameters[name] = torch.as_tensor(array)
            for name, array in tensor_dict["buffers"].items():
                module.register_buffer(name, torch.as_tensor(array))