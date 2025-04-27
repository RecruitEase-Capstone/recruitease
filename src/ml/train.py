import pandas as pd
import numpy as np
import torch
import json
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    IntervalStrategy
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from seqeval.metrics import classification_report
import evaluate

# Define the tag mapping for our NER task
tag2idx = {
    "O": 0,
    "B-nama": 1,
    "I-nama": 2,
    "B-pendidikan": 3,
    "I-pendidikan": 4,
    "B-pengalaman_kerja": 5,
    "I-pengalaman_kerja": 6,
    "B-keterampilan": 7,
    "I-keterampilan": 8
}

idx2tag = {i: tag for tag, i in tag2idx.items()}

# Function to convert tags to BIO format (Begin, Inside, Outside)
def convert_to_bio_format(tokens_tags):
    bio_tags = []
    prev_tag = "O"
    
    for token, tag in tokens_tags:
        if tag == "O":
            bio_tags.append("O")
        elif tag != prev_tag:
            bio_tags.append(f"B-{tag}")
        else:
            bio_tags.append(f"I-{tag}")
        prev_tag = tag
    
    return list(zip([token for token, _ in tokens_tags], bio_tags))

# Dataset class
class ResumeNERDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=10000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for _, row in tqdm(dataframe.iterrows(), desc="Processing dataset", total=len(dataframe)):
            if pd.isna(row['tag']) or row['tag'] == "":
                continue
                
            # Parse the JSON string containing token-tag pairs
            try:
                tokens_tags = json.loads(row['tag'])
                
                # Convert to BIO format
                bio_tokens_tags = convert_to_bio_format(tokens_tags)
                
                tokens = [item[0] for item in bio_tokens_tags]
                tags = [item[1] for item in bio_tokens_tags]
                
                # Tokenize
                inputs = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                word_ids = inputs.word_ids()
                label_ids = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)  #(ignored by loss function)
                    else:
                        if word_idx < len(tags):
                            # Map the tag to its ID
                            tag = tags[word_idx]
                            label_ids.append(tag2idx.get(tag, 0))
                        else:
                            label_ids.append(0)  # Default to 'O'
                
                # Add to samples
                self.samples.append({
                    'input_ids': inputs['input_ids'][0],
                    'attention_mask': inputs['attention_mask'][0],
                    'labels': torch.tensor(label_ids)
                })
                
            except Exception as e:
                print(f"Error processing row: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index
    true_predictions = [
        [idx2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [idx2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    parser = argparse.ArgumentParser(description="Train NER model based on the tagged Resume data")
    parser.add_argument("--input_file", required=True, help="Path to the CSV file with token tags")
    parser.add_argument("--output_dir", default="./model", help="Directory to save the trained model")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")
    
    # Load the pre-trained tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('dslim/bert-base-NER')
    
    # Save the tag mappings
    with open(os.path.join(args.output_dir, "tag_mapping.json"), "w") as f:
        json.dump({"tag2idx": tag2idx, "idx2tag": {str(i): tag for i, tag in idx2tag.items()}}, f)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ResumeNERDataset(train_df, tokenizer, max_length=args.max_length)
    val_dataset = ResumeNERDataset(val_df, tokenizer, max_length=args.max_length)
    
    # Load the pre-trained model
    model = BertForTokenClassification.from_pretrained(
        'dslim/bert-base-NER',
        num_labels=len(tag2idx),
        id2label=idx2tag,
        label2id=tag2idx
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",          
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f)
    
    print("Training complete!")

if __name__ == "__main__":
    main()