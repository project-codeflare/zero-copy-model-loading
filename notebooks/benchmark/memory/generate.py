import transformers

GENERATE_MODEL_NAME = 'gpt2'
GENERATE_INPUT = {
    'prompt_text': 'All your base are'
}

generate_pipeline = transformers.pipeline(
    'text-generation', model=GENERATE_MODEL_NAME)
pad_token_id = generate_pipeline.tokenizer.eos_token_id

generate_pre = generate_pipeline.preprocess(**GENERATE_INPUT)
generate_output = generate_pipeline.forward(generate_pre,
                                            pad_token_id=pad_token_id)
print(generate_pipeline.postprocess(generate_output))
