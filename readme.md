## Overview of Methods
In this work, we have 4 steps:

**Step 1:** Training Retrieval Model via Document Index

**Step 2:** Query Retrieval Model with Privacy Enhancement

**Step 3:** Resampling & Synthetic Data

**Step 4:** Trustworthy Quantization (TQU)


## Prepared the data:
    data can be download using ./Step_1/download_data.py

Note that prepare the different dataset to change the line of load_xx() in download_data.py.

## How to run step 1:
    We are using sst2 as an example. After download the data using download_data.py, you can use cluster_sst2.py to cluster these public points. Then you can use the clustering label and public data to training retrieval model using train_retrival_model.py

Note that you may need to change the directory path in train_retrival_model.py.
## How to run step 2:
    You can then use query_retrival_model.py to get histogram with DP. You will get the results of resampling. 
Note that you may need to change the directory path in query_retrival_model.py.

## How to run step 3:
    You can then use fine-tune-generator.py to finetune the generator using resampling data. Finally, generate synthetic data. 

## How to run step 4:
    In step 4, you can use train_downstream.py to train new downstream model using resampling + synthetic data. Then using TQU in ./Step_4/TQU






