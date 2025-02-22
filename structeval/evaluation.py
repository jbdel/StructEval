import os
import json
import datetime
import ast
from tqdm import tqdm
from typing import List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report


##########################################################################
# Configurations
##########################################################################
DATA_PATH = "data/test_reviewed.json"
RESULTS_DIR = "./"
MODEL_PATH = "StanfordAIMI/SRRG-BERT-Upper-with-Statuses"

MODE = "upper_with_statuses"  # "leaves", "upper", "leaves_with_statuses", "upper_with_statuses"
MAPPING_FILE = f"{MODE}_mapping.json"

MAX_LENGTH = 128  # Max token length for tokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
##########################################################################


@dataclass
class Sentence:
    """Dataclass to represent a sentence with its corresponding labels."""
    text: str
    raw_labels: List[str] = field(default_factory=list)
    leaves: List[str] = field(default_factory=list)
    upper: List[str] = field(default_factory=list)
    leaves_with_statuses: List[str] = field(default_factory=list)
    upper_with_statuses: List[str] = field(default_factory=list)


class DiseaseDataset(Dataset):
    """Custom dataset class for processing disease-related sentences."""
    def __init__(self, sentences: List[Sentence], tokenizer, label_field, max_length=MAX_LENGTH):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label_field = label_field
        self.max_length = max_length

        # Load label mapping file
        mapping_path = f"{label_field}_mapping.json"
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping file {mapping_path} not found!")

        with open(mapping_path, "r") as f:
            self.label_map = json.load(f)

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoding = self.tokenizer.encode_plus(
            sentence.text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        # Determine label set
        label_list = getattr(sentence, self.label_field, [])
        label_ids = [self.label_map[label] for label in label_list if label in self.label_map]

        # Convert labels to one-hot tensor
        labels_tensor = torch.zeros(len(self.label_map), dtype=torch.float)
        labels_tensor[label_ids] = 1.0
        
        return {
            "sentence": sentence,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels_tensor
        }


def load_data(file_path, mode):
    """Loads data from JSON file and filters based on the specified mode."""
    with open(file_path, "r") as f:
        data = json.load(f)

    # Filter sentences with available upper_with_statuses if in that mode
    if mode == "upper_with_statuses":
        sentences = [Sentence(text=sentence.pop("utterance"), **sentence) for sentence in data if sentence["upper_with_statuses"]]
    else:
        sentences = [Sentence(text=sentence.pop("utterance"), **sentence) for sentence in data]

    return sentences


def load_model(model_path, labels):
    """Loads a pretrained model for classification."""
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(labels))
    model.to(DEVICE)
    return model


def evaluate_model(model, dataset, output_csv_path):
    """Evaluates the model and saves results."""
    model.eval()
    predictions, true_labels, sentences = [], [], []

    with torch.no_grad():
        for data in tqdm(dataset, desc="Evaluating"):
            input_ids = data["input_ids"].unsqueeze(0).to(DEVICE)
            attention_mask = data["attention_mask"].unsqueeze(0).to(DEVICE)
            labels = data["labels"].unsqueeze(0).to(DEVICE)
            sentence = data["sentence"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.sigmoid(logits) > 0.5

            predictions.append(preds.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
            sentences.append(sentence)
    
    # Convert to numpy arrays
    predictions = np.concatenate(predictions, axis=0).astype(int)
    true_labels = np.concatenate(true_labels, axis=0).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="samples")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save results
    output_df = pd.DataFrame({
        "sentence": [s.text for s in sentences],
        "true_labels": true_labels.tolist(),
        "predictions": predictions.tolist()
    })
    return output_df


##########################################################################
# Main
##########################################################################

if __name__ == "__main__":
    # Load label mapping
    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)
        labels = mapping.keys()

    # Load model
    model = load_model(MODEL_PATH, labels=labels)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")

    # Load test data
    sentences = load_data(DATA_PATH, MODE)

    # Create dataset
    test_dataset = DiseaseDataset(sentences, tokenizer, label_field=MODE)

    # Evaluate model
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_csv_path = os.path.join(RESULTS_DIR, f"{timestamp}-{MODE}.csv")
    results = evaluate_model(model, test_dataset, output_csv_path)

    # Save predictions to CSV
    results.to_csv(output_csv_path, index=False)

    y_true = results["true_labels"].tolist()
    y_pred = results["predictions"].tolist()

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=labels)

    # Save classification report
    report_file = os.path.join(RESULTS_DIR, f"{timestamp}-{MODE}_classification_report.txt")
    with open(report_file, "w") as f:
        f.write(report)

    print(f"Classification report saved to: {report_file}")
