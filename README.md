# Environment Setup

1. Run `pip install -r requirements.txt` to install the Python dependencies.  
2. Prepare your OpenAI API key and add it to the `GPT_4O_MINI_KEY` field in the `env.env` file.  


# Deploy ConceptNet

1. Download the ConceptNet assertions from the [ConceptNet official website](https://github.com/commonsense/conceptnet5/wiki/Downloads).  
2. Run `utils/conceptnet/to_neo4j.py` to convert the assertions into a format suitable for importing into Neo4j, generating `nodes.csv` and `edges.csv`.  
3. Download Neo4j from the [Neo4j official website](https://neo4j.com/deployment-center/), deploy it, and import `nodes.csv` and `edges.csv` to construct ConceptNet.  
4. Run `utils/conceptnet/vecdb.py` to generate embeddings for each node and build a vector database (make sure to set the relevant parameters and variables at the end of the Python script).  


# Preprocess Dataset

1. Download the datasets:  
- [Gao's](https://github.com/ArrogantL/ChatGPT4CausalReasoning)  
- [CPATT](https://github.com/NLPCodebase/Code4CPATT)  
2. Preprocess the datasets by adding necessary fields required for subsequent steps. You can use `CPATT_dataset_format.ipynb` to preprocess the CPATT dataset by specifying the `split` and `subset_type` for each dataset. (Gao’s dataset can be processed similarly; just ensure the fields are consistent.)  
3. Run `utils/dataset_preprocess.py` to process the data (make sure to set the relevant parameters and variables at the beginning and end of the Python script).  


## Inference

Run `sere.py` to perform inference (make sure to set the relevant parameters and variables at the end of the script).
