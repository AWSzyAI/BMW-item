# from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
# print(type(tokenizer))
# model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-chinese")
# print(type(model))

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
print(type(tokenizer))
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-chinese")
print(type(model))