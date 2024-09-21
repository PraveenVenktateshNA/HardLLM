

import json
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm

tokenizer = MT5Tokenizer.from_pretrained('./models/sst2-cluster-mt5-large-DSI/checkpoint-140000')
model = MT5ForConditionalGeneration.from_pretrained('./models/sst2-cluster-mt5-large-DSI/checkpoint-140000')

model.eval()

# define SPIECE_UNDERLINE 和 INT_TOKEN_IDS
SPIECE_UNDERLINE = "▁"
INT_TOKEN_IDS = []
for token, id in tokenizer.get_vocab().items():
    if token.startswith(SPIECE_UNDERLINE) and token[1:].isdigit():
        INT_TOKEN_IDS.append(id)
    elif token == SPIECE_UNDERLINE or token.isdigit():
        INT_TOKEN_IDS.append(id)
INT_TOKEN_IDS.append(tokenizer.eos_token_id)

def restrict_decode_vocab(batch_idx, prefix_beam):
    return INT_TOKEN_IDS

# load private dataset
input_file = './data/sst2/sst_private.jsonl'
with open(input_file, 'r') as f:
    lines = f.readlines()

sentences = [json.loads(line)['sentence'] for line in lines]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

decoded_outputs_list = []

batch_size = 32
num_batches = len(sentences) // batch_size + (1 if len(sentences) % batch_size != 0 else 0)


for i in tqdm(range(num_batches), desc="Processing sentences in batches"):
    batch_sentences = sentences[i * batch_size: (i + 1) * batch_size]
    input_ids = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            max_length=50, 
            num_beams=20, 
            prefix_allowed_tokens_fn=restrict_decode_vocab, 
            num_return_sequences=5,  # modify this based on requirements
            early_stopping=True
        )
        for j in range(len(batch_sentences)):
            outputs_list = []
            for k in range(5):  # num_return_sequences
                decoded_outputs = tokenizer.decode(outputs[j * 5 + k], skip_special_tokens=True)
                outputs_list.append(decoded_outputs)
            decoded_outputs_list.append(outputs_list)
output_dict = defaultdict(int)
flattened_list = [item for sublist in decoded_outputs_list for item in sublist]
for item in flattened_list:
    output_dict[item] += 1

# decoded_outputs_list format : {"001":20, "002":30}

# add noise
def add_gaussian_noise(counts, sensitivity, epsilon, delta):

    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    noisy_counts = {}
    
    for key, count in counts.items():
        noise = np.random.normal(0, sigma)
        noisy_count = int(round(count + noise))
        noisy_counts[key] = max(0, noisy_count) 
    
    return noisy_counts
sensitivity = 1
epsilon = 1 
delta = 0.01 
noisy_output_dict = add_gaussian_noise(output_dict, sensitivity, epsilon, delta)
noisy_output_dict = {k: v for k, v in noisy_output_dict.items() if v != 0}
print(noisy_output_dict)

noisy_output_file = './data/sst2/noisy_output.jsonl'
noisy_output_dict = {}

with open(noisy_output_file, 'r') as f:
    for line in f:
        json_obj = json.loads(line)
        noisy_output_dict.update(json_obj)

def compute_p(d, D):
    N = len(D)
    numerator = np.sum([np.linalg.norm(d - di) for di in D])
    denominator = np.sum([np.linalg.norm(di - dj) for i, di in enumerate(D) for dj in D[i+1:]])
    return 1 - numerator / denominator
save_directory = "./data/sst2/sst2_with_clusters.jsonl"
dataset = load_dataset('json', data_files=save_directory)
train_data = dataset['train']
model_sentence = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model_sentence.max_seq_length = 50  
all_sampled_data = []
for k, v in tqdm(noisy_output_dict.items(), desc="Processing dictionary items"):
    train_data_clustering = []
    for item in dataset["train"]:
        if str(item["text_id"]) == k:
            train_data_clustering.append(item)
    sentences = [item['sentence'] for item in train_data_clustering]
    embeddings = model_sentence.encode(sentences,batch_size=64,show_progress_bar=True)
    if v >= len(train_data_clustering):
        sampled_data = train_data_clustering
    else:
        probabilities = []
        for d in embeddings:
            D = [di for di in embeddings if not np.array_equal(di, d)]
            p = compute_p(d, D)
            probabilities.append(p)
        # Assign probabilities back to the data points
        for item, p in zip(train_data_clustering, probabilities):
            item['probability'] = p
        sampled_indices = np.random.choice(len(train_data_clustering), size=v, replace=False, p=probabilities/np.sum(probabilities))
        sampled_data = [train_data_clustering[i] for i in sampled_indices]
    all_sampled_data.extend(sampled_data)
output_file = './data/sst2/selected.jsonl'
with open(output_file, 'w') as f:
    for item in all_sampled_data:
        f.write(json.dumps(item) + '\n')
print(f"Sampled data has been written to {output_file}")


    
        
