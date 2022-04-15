import transformers
import torch

text =  "We're not happy unless you're not happy."

SENTIMENT_MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment'
sentiment_tokenizer = transformers.AutoTokenizer.from_pretrained(
    SENTIMENT_MODEL_NAME)
sentiment_model = (
    transformers.AutoModelForSequenceClassification
    .from_pretrained(SENTIMENT_MODEL_NAME))

model_input = sentiment_tokenizer(text, padding=True, 
                                  return_tensors='pt')
print(sentiment_model(**model_input))


