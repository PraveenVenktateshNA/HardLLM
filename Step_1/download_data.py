from datasets import load_dataset, DatasetDict, concatenate_datasets, ClassLabel 
import random 
import json 
def add_source_label(example): 
    example['source'] = 'correct' 
    return example 
def add_source_label_amazon(example): 
    example['source'] = 'incorrect' 
    return example 


def map_amazon_label(example): 
    if example['label'] in [0, 1,2]: 
        example['label'] = 0 
    elif example['label'] in [3, 4]: 
        example['label'] = 1 
        return example 


def load_sst2():
    dataset_amazon = load_dataset("SetFit/amazon_reviews_multi_en") 
    dataset_sst2 = load_dataset("glue", "sst2")
    dataset_amazon = dataset_amazon.remove_columns("id") 
    dataset_amazon = dataset_amazon.remove_columns("label_text") 
    dataset_amazon = dataset_amazon.rename_column("text", "sentence") 
    dataset_sst2 = dataset_sst2.cast_column("label", ClassLabel(names=["negative", "positive"])) 
    dataset_amazon = dataset_amazon.cast_column("label", ClassLabel(names=["negative", "positive"])) 
    dataset_sst2['train'] = dataset_sst2['train'].remove_columns('idx')  
    dataset_sst2 = dataset_sst2.cast_column("label", ClassLabel(names=["negative", "positive"])) 
    sst2_train = dataset_sst2['train'].shuffle(seed=42) 
    dataset_amazon['train'] = dataset_amazon['train'].map(map_amazon_label) 
    sst2_train = sst2_train.map(add_source_label)
    dataset_amazon['train'] = dataset_amazon['train'].map(add_source_label_amazon)
    split_index = int(0.5 * len(sst2_train)) 
    sst2_train_public = sst2_train.select(range(split_index)) 
    sst2_train_private = sst2_train.select(range(split_index, len(sst2_train)))
    combined_dataset = concatenate_datasets([sst2_train_public, dataset_amazon['train']])
    with open('./sst2/sst_public.jsonl', 'w') as f: 
        for example in combined_dataset: 
            f.write(json.dumps(example) + '\n') 
    with open('./sst2/sst_private.jsonl', 'w') as f: 
        for example in sst2_train_private: 
            f.write(json.dumps(example) + '\n') 
    print("Finished to restore dataset locally for sst2 experiments")

    
def load_amazon():
    dataset_amazon = load_dataset("SetFit/amazon_reviews_multi_en") 
    dataset_yelp = load_dataset("Yelp/yelp_review_full") 
    dataset_amazon = dataset_amazon.remove_columns("id") 
    dataset_amazon = dataset_amazon.remove_columns("label_text")
    with open('./amazon/amazon_public.jsonl', 'w') as f: 
        for example in dataset_yelp['train']: 
            f.write(json.dumps(example) + '\n')
    with open('./amazon/amazon_private.jsonl', 'w') as f: 
        for example in dataset_amazon['train']: 
            f.write(json.dumps(example) + '\n')
    print("Finished to restore dataset locally for amazon experiments")

def load_dial():
    dataset_diasum = load_dataset("knkarthick/dialogsum") 
    dataset_sam = load_dataset("Samsung/samsum") 
    dataset_diasum_train = dataset_diasum['train'].remove_columns(["id","topic"])
    dataset_sam_train = dataset_sam['train'].remove_columns(["id"]) 
    split_index = int(0.5 * len(dataset_diasum_train)) 
    dataset_diasum_train_public = dataset_diasum_train.select(range(split_index)) 
    dataset_diasum_train_privacy = dataset_diasum_train.select(range(split_index, len(dataset_diasum_train)))
    combined_dataset = concatenate_datasets([dataset_diasum_train_public, dataset_diasum_train]) 
    with open('./dialsum/dialsum_public.jsonl', 'w') as f: 
        for example in combined_dataset: 
            f.write(json.dumps(example) + '\n') 
    with open('./dialsum/dialsum_private.jsonl', 'w') as f:
        for example in dataset_diasum_train_privacy: 
            f.write(json.dumps(example) + '\n') 
    print("Finished to restore dataset locally for dial experiments")

# change this line to download different datasets
load_sst2()





