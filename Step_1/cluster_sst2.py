from sklearn.cluster import MiniBatchKMeans 
from sentence_transformers import SentenceTransformer 
from datasets import load_dataset 
from sklearn.decomposition import IncrementalPCA 
from tqdm import tqdm 
import gc 
import numpy as np 
import os 
import json 

# change the name of dir for different dataset
save_directory = "./sst2/sst_public.jsonl" 
dataset = load_dataset('json', data_files=save_directory) 
print("ADD Dataset....") 
train_data = dataset['train'] 
sentences = train_data['sentence'] 
print("Generate Embeddings....") 
# Load a pre-trained model 
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 
# Convert sentences to embeddings 
embeddings = model.encode(sentences,batch_size=64,show_progress_bar=True) 
# Reduce dimensionality with Incremental PCA 
batch_size = 500 # You can adjust this batch size according to your memory limits 
ipca = IncrementalPCA(n_components=48, batch_size=batch_size) 
for i in tqdm(range(0, embeddings.shape[0], batch_size)): 
    ipca.partial_fit(embeddings[i:i+batch_size]) 
    gc.collect() 
reduced_embeddings = ipca.transform(embeddings) 
print("Clustering....") 
num_clusters = len(sentences) // 100 
kmeans = MiniBatchKMeans( 
    n_clusters=num_clusters, 
    init='k-means++', 
    max_iter=100, 
    batch_size=50, 
    verbose=0, 
    compute_labels=True, 
    random_state=0, 
    tol=0.0,
    max_no_improvement=10, 
    init_size=None,
    n_init=3,
    reassignment_ratio=0.01 
) 
model = kmeans.fit(reduced_embeddings)
print("Generate Clustering Label....") 
labels = kmeans.labels_ 
# Add cluster labels to the original data 
new_data = [] 
for idx, label in enumerate(labels): 
    entry = train_data[idx].copy() 
    entry['text_id'] = int(label) 
new_data.append(entry) 
new_save_directory = os.path.join('./sst2/', 'sst2_with_clusters.jsonl') 
with open(new_save_directory, 'w') as f: 
    for entry in new_data: 
        f.write(json.dumps(entry) + '\n') 
print(f"Data with cluster labels saved to {new_save_directory}") 

 