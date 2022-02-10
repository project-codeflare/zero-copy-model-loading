#
#  Copyright (c) 2022 IBM Corp.
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

'''
Functions for invoking models that have been rewritten for zero-copy 
loading.
'''

import copy
import ray
import torch
import zerocopy.rewrite

from typing import Any


@ray.remote
def call_model(model_ref: ray.ObjectRef,
               args: Any = (),
               kwargs: Any = None,
               method_name: str = '__call__') -> Any:
    '''
    Ray task that uses zero-copy model loading to reconstitute a model
    from Plasma, then a method on the model.

    :param model_ref: Object reference to a tuple of model skeleton
     and model weights, as returned by :func:`extract_tensors`
    :param args: Ordered arguments to pass to the model's method
    :param kwargs: Keyword arguments to pass to the model's method,
                   or `None` to pass no keyword arguments
    :param method_name: Name of the method to call on the object

    :returns: Return value from calling the specified method
    '''
    if kwargs is None:
        kwargs = {}

    # Suppress PyTorch warnings about immutable tensors
    import warnings
    warnings.filterwarnings('ignore')

    model_skeleton, model_weights = model_ref
    zerocopy.rewrite.replace_tensors(model_skeleton, model_weights)
    with torch.no_grad():
        method = getattr(model_skeleton, method_name)
        return method(*args, **kwargs)
    
def rewrite_pipeline(pipeline: Any, method_names = ('__call__',)) -> Any:
    '''
    Rewrites PyTorch models in a model processing pipeline into Ray tasks
    that load the model using zero-copy model loading.
    
    Current limitatations:
    * Only models that are stored in fields of the top-level object will be
      rewritten. This method does *not* recursively traverse child objects.
    * Only models that are subclasses of ``torch.nn.Module`` will be rewritten.
    * If there are multiple pointers to the same model, they will be 
      treated as separate models and loaded separately onto Plasma.
    * ``pipeline`` must be an object that will still work properly after
      being shallow-copied with :func:`copy.copy()`

    :param pipeline: Python object that wraps a model serving pipeline
    :param method_names: Names of model methods to forward to remote classes.
    
    :returns: A **shallow** copy of ``pipeline`` in which all PyTorch models
     that are stored in fields of ``pipeline`` are replaced with wrapper
     objects that forward calls to Ray tasks.
    '''
    # Find all the models hanging directly off of the pipeline object.
    model_attr_names = [name for name in dir(pipeline) 
                        if isinstance(getattr(pipeline, name), torch.nn.Module)]
    
    # Shallow-copy the original pipeline
    result = copy.copy(pipeline)
    
    # Replace models with shims that push model inference onto Ray tasks.
    for name in model_attr_names:
        model = getattr(result, name)
        model_ref = ray.put(zerocopy.rewrite.extract_tensors(model))
        
        # Define a separate class for each shim.
        class _Callback:
            _model_ref = model_ref
        
        # Generate a class method for each method we forward
        for method_name in method_names:
            if hasattr(model, method_name):
                def method_shim(cls_, *args, **kwargs):
                    return ray.get(zerocopy.call_model.remote(cls_._model_ref, 
                                                     args, kwargs,
                                                     method_name))
                
                setattr(_Callback, method_name, method_shim)
            
        setattr(result, name, _Callback())
    
    return result
    