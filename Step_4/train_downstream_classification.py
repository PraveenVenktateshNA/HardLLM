from datasets import load_dataset, DatasetDict, Dataset,ClassLabel

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
from torch.nn.functional import cross_entropy

save_directory = "/home/zehang/Swinburne_work/Security/compressed_dp/DSI-QG-main/data/sst2/selected_ep2.jsonl"

dataset = load_dataset('json', data_files=save_directory)
def convert_label_to_int(example):
    example['labels'] = int(example['labels'])
    return example
print("ADD Dataset....")
train_data = dataset['train']
train_data = train_data.remove_columns(['source', 'text_id', 'probability'])
# train_data = train_data.remove_columns(['source'])
train_data = train_data.rename_column("label","labels")
# train_data = train_data.map(convert_label_to_int)
# sst2
# The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. It uses the two-way (positive/negative) class split, with only sentence-level labels.
dataset = load_dataset("glue", "sst2")
dataset_test = dataset['test']
dataset_test = dataset_test.remove_columns(['idx'])
dataset_test = dataset_test.rename_column("label","labels")


dataset_valid = dataset['validation']
dataset_valid = dataset_valid.remove_columns(['idx'])
dataset_valid = dataset_valid.rename_column("label","labels")




model_checkpoint = 'roberta-base'

# define label maps
id2label = {0: "negative", 1: "positive"}
label2id = {"negative":0, "positive":1}

# generate classification model from model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

## preprocess data

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["sentence"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=512
    )

    return tokenized_inputs
# tokenize training and validation datasets
tokenized_train_data = train_data.map(tokenize_function, batched=True)
tokenized_test_data = dataset_test.map(tokenize_function, batched=True)
tokenized_valid_data = dataset_valid.map(tokenize_function, batched=True)



# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}



peft_config = LoraConfig(task_type="qdp",
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ['query'])
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# hyperparameters
lr = 1e-3
batch_size = 16
num_epochs = 20


training_args = TrainingArguments(
    output_dir= model_checkpoint + "-lora-text-classification-sst2-ep2-modified",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    label_names=["labels"]
    
)

# Define training arguments
# training_args = TrainingArguments(
#     output_dir=model_checkpoint + "-lora-text-classification-ep1",
#     learning_rate=lr,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=0.01,
#     weight_decay=0.01,
#     evaluation_strategy="steps",
#     save_strategy="steps",
#     eval_steps=10,  # Evaluate every 10 steps
#     save_steps=10,  # Save checkpoint every 10 steps
#     label_names=["labels"],
# )

# creater trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_valid_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
if torch.cuda.is_available():
    model.cuda()

# train model
trainer.train()

# model.to('cpu')
# correct_predictions = 0
# for text, true_label in zip(dataset_valid['sentence'], dataset_valid['labels']):
#     # Tokenize the text and prepare input tensor
#     inputs = tokenizer.encode(text, return_tensors="pt").to("cpu")
    
#     # Get predictions from the model
#     with torch.no_grad():  # Inference mode, no need to calculate gradients
#         logits = model(inputs).logits
    
#     # Determine the predicted label
#     predictions = torch.max(logits, dim=1).indices.item()  # Using .item() to get a Python number
    
#     # Increment correct predictions counter if prediction is correct
#     if predictions == true_label:
#         correct_predictions += 1
# # Calculate accuracy
# accuracy = correct_predictions / len(dataset['validation']['sentence'])
# print("Accuracy:", accuracy)

# save_directory = "./models/trained_model-fullsst2"  # Specify your save directory
# trainer.save_model(save_directory)
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)
     
     

