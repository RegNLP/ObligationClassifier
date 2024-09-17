
# Obligation Classificiation Data

This dataset was built using the Abu Dhabi Global Market (ADGM) Financial Regulations, leveraging few-shot learning with the `gpt-4-turbo-1106-246` model.


# Obligation Classification using LegalBERT

This project involves fine-tuning the LegalBERT model to classify text data into two categories: obligations (True) and non-obligations (False). The text data is provided in a JSON format and tokenized using the LegalBERT tokenizer. The script leverages PyTorch, the Hugging Face Transformers library, and Scikit-learn for model training, evaluation, and performance metrics.

## Steps in the Code

### 1. **Data Loading and Preprocessing**
- The input data is read from a JSON file, where each entry consists of a "Text" field and an "Obligation" field.
- Texts are extracted from the JSON, and the `Obligation` field is converted into binary labels (1 for obligation, 0 for non-obligation).

### 2. **Tokenization**
- LegalBERT's tokenizer (`nlpaueb/legal-bert-base-uncased`) is used to tokenize the text data. Special tokens, padding, and attention masks are applied to prepare the data for the model.

### 3. **Dataset Preparation**
- A custom `Dataset` class (`ObligationDataset`) is created to handle the tokenized inputs. The data is split into training and validation sets using `train_test_split`.

| Total Items | Obligation:True | Obligation: False |
|-------------|-----------------|-------------------|
| 2296        | 1414            | 882               |


### 4. **Model Fine-Tuning**
- The pre-trained LegalBERT model is fine-tuned for sequence classification with two output labels (obligation or non-obligation).
- The `Trainer` class from Hugging Face's Transformers library is used to train the model. 
- Training arguments such as batch size, learning rate scheduling, weight decay, and early stopping are defined.

### 5. **Evaluation**
- The model is evaluated on validation data using metrics such as accuracy, precision, recall, and F1-score, computed via `precision_recall_fscore_support` and `accuracy_score`.

### 6. **Saving the Model**
- After training, the model and tokenizer are saved for future use in the `./obligation-classifier-legalbert` directory.

## Requirements

- Python 3.x
- PyTorch
- Hugging Face Transformers
- Scikit-learn

## Running the Code

1. Install the required libraries:
   ```bash
   pip install torch transformers scikit-learn

2. Place the JSON file (ObligationClassificationDataset.json) in the appropriate directory.

3. Run the script:
```
    python obligation_classification.py
```


4. After training, the model and tokenizer will be saved to ./obligation-classifier-legalbert.

### BibTeX

```shell
@misc{gokhan2024regnlpactionfacilitatingcompliance,
      title={RegNLP in Action: Facilitating Compliance Through Automated Information Retrieval and Answer Generation}, 
      author={Tuba Gokhan and Kexin Wang and Iryna Gurevych and Ted Briscoe},
      year={2024},
      eprint={2409.05677},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.05677}, 
}
