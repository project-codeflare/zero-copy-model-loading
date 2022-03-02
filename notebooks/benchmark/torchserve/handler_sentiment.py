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


class SentimentHandler(BaseHandler):
    '''
    Handler for the Sentiment model in our benchmark.
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
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        self.model = (transformers.AutoModelForSequenceClassification
                      .from_pretrained(model_dir))

        self.model.eval()

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
                    Just kidding, we return a pair of tensors.
        '''
        contexts = [_get_input_record(request, ('context',))['context']
                    for request in data]

        return self.tokenizer(contexts, padding=True, return_tensors='pt')

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
            Just kidding, this is actually a complex data structure.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this 
                           function.
                           Just kidding, the returned value is actually a 
                           SequenceClassifierOutput object.
        '''
        return self.model(**input_batch)

    def postprocess(self, inference_output):
        '''
        Overridden version of the eponymous method in the base class.

        Description from the base class:

        *The post process function makes use of the output from the inference 
        and converts into a Torchserve supported response output.*

        Args:
            data (Torch Tensor): The torch tensor received from the prediction
            output of the model.
            Just kidding, the input is actually a SequenceClassifierOutput 
            object.
        Returns:
            List: The post process function returns a list of the predicted 
            output.
        '''
        # The `logits` field of the input SequenceClassifierOutput object
        # contains a 2d array of logits.
        scores = inference_output.logits.detach().numpy()
        scores = scipy.special.softmax(scores, axis=1).tolist()
        scores = [{k: v for k, v 
                   in zip(['positive', 'neutral', 'negative'], row)}
                  for row in scores]
        return scores

