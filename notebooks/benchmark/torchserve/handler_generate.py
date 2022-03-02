import json
import logging

from typing import Any, List

from ts.torch_handler.base_handler import BaseHandler
import scipy.special
import transformers


logger = logging.getLogger(__name__)


def _get_input_record(request: Any, field_names: List[str]) -> str:
    '''
    Helper function to pull out input strings from the ``data`` object
    that TorchServe provides you with.

    Args:
        request: A single request
        field_names: The names of the field to extract from the request.
    '''
    # By convention request should be a JSON record, containing the specified
    # fields, and maybe some other fields.
    #
    # TorchServe helpfully gives you the raw binary data of the request.
    # Drill down to the requested parts and decode strings as needed.
    input_json = json.loads(request['body'].decode('utf-8'))

    if not isinstance(input_json, dict):
        raise ValueError(f'Request data {request} of type {type(request)} '
                         f'is not a JSON record.')
    for key in field_names:
        if key not in input_json:
            raise ValueError(f'Request data {request} of type {type(request)} '
                             f'does not contain required key "{key}"')
    return {key: input_json[key] for key in field_names}

class GenerateHandler(BaseHandler):
    '''
    Handler for the natural language generation model in our benchmark.
    '''

    def initialize(self, context):
        '''
        Overridden version of the eponymous method in the base class.

        Description from the base class:

        *Initialize function loads the model.pt file and initialized the model
        object. First try to load torchscript else load eager mode state_dict 
        based model.*

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing
        '''

        # Serialized model is buried behind several levels of indirection.
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get('model_dir')

        # Use Huggingface's loading code instead of calling Module.load_state_dict()
        # like the base class does.
        # To load a pipeline saved with save_pretrained(), you pass the location 
        # of the directory as a the `model` argument to `pipeline()`.
        self.pipeline = transformers.pipeline(
            'text-generation', model=model_dir)
        self.pipeline.model.eval()  # Just in case
        self.pad_token_id = self.pipeline.tokenizer.eos_token_id

        logger.info(
            f'Transformer model from path {model_dir} loaded successfully.'
        )

        self.initialized = True

    def preprocess(self, data):
        '''
        Overridden version of the eponymous method in the base class.

        Description from the base class:

        *Preprocess function to convert the request input to a tensor 
        (Torchserve supported format).
        The user needs to override to customize the pre-processing.*

        Args:
            data (list): List of the data from the request input.
        Returns:
            tensor: Returns the tensor data of the input
                    Just kidding, we return a list of dictionaries of
                    tensors and other stuff.
        '''
        # Expected input format:
        # [ { 'prompt_text': 'All your base are' }, ... ]
        # where the outer list is `data` and the inner elements are raw request
        # objects.
        
        # Start by parsing the JSON in each request.
        json_records = [
            _get_input_record(request, ('prompt_text',))
            for request in data
        ]
        
        # preprocess() takes a single input at a time, but we need to handle 
        # a batch at a time.
        return [self.pipeline.preprocess(**r) for r in json_records]

    def inference(self, input_batch):
        '''
        Overridden version of the eponymous method in the base class.

        Description from the base class:

        *The Inference Function is used to make a prediction call on the given
        input request. The user needs to override the inference function to 
        customize it.*

        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference
            Request.
            The shape should match the model input shape.
            Just kidding, this is actually a list of dictionaries.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
            Just kidding, we return another list of dictionaries.
        '''
        # forward() takes a single input at a time, but we need to run a
        # batch at a time.
        return [self.pipeline.forward(r, pad_token_id=self.pad_token_id)
                for r in input_batch]

    def postprocess(self, inference_output):
        '''
        Overridden version of the eponymous method in the base class.

        Description from the base class:

        *The post process function makes use of the output from the inference 
        and converts into a Torchserve supported response output.*

        Args:
            data (Torch Tensor): The torch tensor received from the prediction
            output of the model.
            Just kidding, this is actually a list of dictionaries.
        Returns:
            List: The post process function returns a list of the predicted 
            output.
        '''
        # postprocess() takes a single generation result at a time, but we
        # need to run a batch at a time.
        return [self.pipeline.postprocess(i) for i in inference_output]
    
