import json
import logging

from ts.torch_handler.base_handler import BaseHandler
import transformers


logger = logging.getLogger(__name__)


def _get_input_text(request) -> str:
    '''
    Helper function to pull out input strings from the ``data`` object
    that TorchServe provides you with.

    Args:
        request: A single request
    '''
    # By convention, we use the following format:
    #
    # intent_input = {
    #     'context':
    #         ("I came here to eat chips and kick butt, "
    #          "and I'm all out of chips.")
    # }
    #
    # TorchServe helpfully decodes this JSON data into Python object and
    # extra-helpfully leaves the strings as binary data.
    # Drill down to the 'context' part and decode.
    input_json = json.loads(request['body'].decode('utf-8'))
    if not isinstance(input_json, dict) or 'context' not in input_json.keys():
        raise ValueError(f'Reqeust data {request} of type {type(request)} '
                         f'does not contain required key "context"')
    input_text = input_json['context']
    if isinstance(input_text, (bytes, bytearray)):
        input_text = input_text.decode('utf-8')
    if input_text is None:
        raise ValueError(f'Couldn\'t get input text from {request}')
    return input_text


class IntentHandler(BaseHandler):
    '''
    Handler for the Intent model in our benchmark.
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
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_dir)

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
        # Ignore the contract specified in the base class and build up complex
        # data structure. The Huggingface-provided example code does this too.
        input_text_list = [_get_input_text(request) for request in data]

        return self.tokenizer(input_text_list, padding=True, return_tensors='pt')

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
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        '''
        return self.model.generate(**input_batch)

    def postprocess(self, inference_output):
        '''
        Overridden version of the eponymous method in the base class.

        Description from the base class:

        *The post process function makes use of the output from the inference 
        and converts into a Torchserve supported response output.*

        Args:
            data (Torch Tensor): The torch tensor received from the prediction
            output of the model.
        Returns:
            List: The post process function returns a list of the predicted 
            output.
        '''
        # Start by producing a list of output strings.
        output_strings = [self.tokenizer.decode(tensor)
                          for tensor in inference_output]

        # Strip the padding tokens. Pity there isn't an option to do this in
        # `decode()`.
        output_strings = [s.replace('<pad>', '') for s in output_strings]

        # Remove the extra space at the beginning and the '</s>' at the end
        output_strings = [s[len(' '):-len('</s>')] for s in output_strings]

        # Reformat as single-element JSON records.
        # Hopefully this format qualifies as a "Torchserve supported
        # response output".
        return [{'intent': s} for s in output_strings]
