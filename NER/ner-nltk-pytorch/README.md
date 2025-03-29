# Named Entity Recognition (NER) with PyTorch

This project implements a Named Entity Recognition (NER) model using PyTorch. The goal is to identify and classify entities in text data, such as names, organizations, locations, and more.

## Project Structure

```
nlp-ner-project
├── data
│   └── sample_data.csv          # Sample data for training and testing
├── notebooks
│   └── nlp_ner_notebook.ipynb   # Jupyter Notebook for NER implementation
├── src
│   ├── data_preprocessing.py     # Data loading and preprocessing functions
│   ├── model.py                  # NER model architecture
│   └── train.py                  # Training loop for the NER model
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Installation

To set up the environment, you need to install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Place your training and testing data in the `data/sample_data.csv` file. 
   - Ensure that it contains text data along with corresponding entity labels.

2. **Run the Notebook**: Open the `notebooks/nlp_ner_notebook.ipynb` file in Jupyter Notebook. 
   This notebook includes sections for:
   - Loading and preprocessing the data
   - Training the NER model
   - Evaluating the model performance

3. **Training the Model**: The training process is handled in the `src/train.py` file, where you can customize the training parameters as needed.
