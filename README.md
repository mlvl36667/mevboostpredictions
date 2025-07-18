# Predictions in MEV-Boost Auctions with Machine Learning
This repository contains the source code to train and evaluate ML models for predictions in MEV-Boost auctions. These results will be (were) presented at BCCA 2025, Dubrovnik.

# Data Avability
Download the raw data from https://zenodo.org/records/15789978

# Data Structure
The PBS auction data must be available at ../ in block_*.json files e.g. block_22246730.json. This is going to process 150.000 blocks, you can track the progress in terminal. This might take a while!

# Preprocessing
Process the raw data using the Python scripts. Run `python3 processor.py` to extract the relevant data to then train ML models.

# Training
Train the ML models using the Python scripts

# Evaluation
Evaluation can be done using the Python scripts from the repository
