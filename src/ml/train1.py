import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForTokenClassification, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from utils import trim_entity_spans, convert_goldparse, ResumeDataset, tag2idx, idx2tag, get_hyperparameters, train_and_val_model, process_resume, evaluate_model_with_report
from sklearn.metrics import classification_report

def custom_collate_fn(batch):
    # Find max lengths for padding
    max_input_len = max([x['input_ids'].size(0) for x in batch])
    max_labels_len = max([x['labels'].size(0) for x in batch])
    
    # Initialize tensors
    input_ids = torch.zeros((len(batch), max_input_len), dtype=torch.long)
    token_type_ids = torch.zeros((len(batch), max_input_len), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_input_len), dtype=torch.long)
    labels = torch.zeros((len(batch), max_labels_len), dtype=torch.long)
    
    # Fill tensors with padded data
    orig_labels = []
    for i, item in enumerate(batch):
        input_len = item['input_ids'].size(0)
        label_len = item['labels'].size(0)
        
        input_ids[i, :input_len] = item['input_ids']
        token_type_ids[i, :input_len] = item['token_type_ids']
        attention_mask[i, :input_len] = item['attention_mask']
        labels[i, :label_len] = item['labels']
        orig_labels.append(item['orig_label'])
    
    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'orig_label': orig_labels
    }

# Function to visualize loss and accuracy
def visualize_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_path):
    """
    Creates visualization of loss and accuracy during training and validation
    
    Args:
        train_losses (list): List of loss values for each epoch during training
        val_losses (list): List of loss values for each epoch during validation
        train_accuracies (list): List of accuracy values for each epoch during training
        val_accuracies (list): List of accuracy values for each epoch during validation
        output_path (str): Path to save the visualization image
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Plot for Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot for Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/training_metrics.png')
    plt.close()
    
    print(f"Training metrics visualization saved at {output_path}/training_metrics.png")

# Function to save classification report to file
def save_classification_report(report, output_path):
    """
    Saves classification report to a text file
    
    Args:
        report (str): Classification report in string format
        output_path (str): Path to save the report file
    """
    with open(f'{output_path}/classification_report.txt', 'w') as f:
        f.write(report)
    print(f"Classification report saved at {output_path}/classification_report.txt")

# Function to visualize token classification predictions
def visualize_token_predictions(text, true_labels, pred_labels, idx2tag, output_path, filename="token_predictions.png"):
    """
    Visualizes token classification predictions vs ground truth
    
    Args:
        text (list): List of tokens
        true_labels (list): List of true token labels
        pred_labels (list): List of predicted token labels
        idx2tag (dict): Dictionary mapping label indices to tag names
        output_path (str): Path to save the visualization
        filename (str): Name of the output file
    """
    # Convert label indices to tag names
    true_tags = [idx2tag[label] if label != -100 else "PAD" for label in true_labels]
    pred_tags = [idx2tag[label] if label != -100 else "PAD" for label in pred_labels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot tokens with color coding based on correct/incorrect predictions
    colors = []
    for true, pred in zip(true_tags, pred_tags):
        if true == "PAD" or true == "O":
            colors.append('lightgray')
        elif true == pred:
            colors.append('lightgreen')
        else:
            colors.append('salmon')
    
    # Display tokens and their predictions
    max_display = min(50, len(text))  # Limit display to avoid overcrowding
    ax.bar(range(max_display), [1] * max_display, color=colors[:max_display])
    
    # Add text labels
    for i in range(max_display):
        if true_tags[i] != "PAD":
            ax.text(i, 1.1, text[i], ha='center', rotation=45)
            ax.text(i, 0.5, f"T: {true_tags[i]}", ha='center', fontsize=8)
            ax.text(i, 0.3, f"P: {pred_tags[i]}", ha='center', fontsize=8)
    
    ax.set_ylim(0, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Token Classification Visualization (Green: Correct, Red: Incorrect, Gray: Padding/Outside)")
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/{filename}')
    plt.close()
    
    print(f"Token prediction visualization saved at {output_path}/{filename}")

class TokenClassificationModel(torch.nn.Module):
    """
    Custom token classification model with BERT base
    """
    def __init__(self, num_labels, pretrained_model_name='bert-base-uncased', dropout_rate=0.3):
        super(TokenClassificationModel, self).__init__()
        
        # Load pretrained BERT model
        self.bert = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # Forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        return outputs


class ModifiedResumeDataset(ResumeDataset):
    def __getitem__(self, idx):
        data = process_resume(
            self.resume[idx], self.tokenizer, self.tag2idx, self.max_len, self.is_test)
        return {
            'input_ids': torch.tensor(data['input_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(data['token_type_ids'], dtype=torch.long) 
                if 'token_type_ids' in data else torch.zeros(len(data['input_ids']), dtype=torch.long),
            'attention_mask': torch.tensor(data['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(data['labels'], dtype=torch.long) if 'labels' in data else torch.zeros(0),
            'orig_label': data.get('orig_labels', [])
        }


def main():
    parser = argparse.ArgumentParser(description='Train Token Classification Model')
    parser.add_argument('-e', type=int, default=1, help='number of epochs')
    parser.add_argument('-o', type=str, default='.', help='output path to save model state')
    parser.add_argument('--model_name', type=str, default='dslim/bert-base-NER', help='pretrained model name')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--lr', type=float, default=1.5e-5, help='learning rate')
    args = parser.parse_args()
    
    output_path = args.o
    
    MAX_LEN = 500
    EPOCHS = args.e
    MAX_GRAD_NORM = 1.0
    MODEL_NAME = args.model_name
    TOKENIZER = BertTokenizerFast('./model/vocab.txt', lowercase=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    
    # Load and prepare data
    data = trim_entity_spans(convert_goldparse('dataset/github/Resumes.json'))
    train_data, val_data = data[:180], data[180:]
    
    # Create datasets
    train_dataset = ModifiedResumeDataset(train_data, TOKENIZER, tag2idx, MAX_LEN)
    val_dataset = ModifiedResumeDataset(val_data, TOKENIZER, tag2idx, MAX_LEN)
    
    # Create dataloaders
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=custom_collate_fn)
    
    # Initialize the token classification model
    model = TokenClassificationModel(
        num_labels=len(tag2idx),
        pretrained_model_name=MODEL_NAME,
        dropout_rate=args.dropout
    )
    model.to(DEVICE)
    
    # Prepare optimizer with hyperparameters
    optimizer_grouped_parameters = get_hyperparameters(model, True)
    optimizer = Adam(optimizer_grouped_parameters, lr=args.lr)
    
    # Train and validate the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_val_model(
        model,
        TOKENIZER,
        optimizer,
        EPOCHS,
        idx2tag,
        tag2idx,
        MAX_GRAD_NORM,
        DEVICE,
        train_dataloader,
        val_dataloader
    )
    
    # Visualize metrics
    visualize_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_path)
    
    # Evaluate model on validation data and get classification report
    print("Evaluating model and creating classification report...")
    eval_results = evaluate_model_with_report(model, val_dataloader, idx2tag, tag2idx, DEVICE)
    
    # Print classification report
    print("\nClassification Report:")
    print(eval_results['classification_report'])
    
    # Save classification report to file
    save_classification_report(eval_results['classification_report'], output_path)
    
    # Visualize token predictions for a sample from validation data
    if hasattr(eval_results, 'sample_tokens') and hasattr(eval_results, 'sample_true_labels') and hasattr(eval_results, 'sample_pred_labels'):
        visualize_token_predictions(
            eval_results.sample_tokens,
            eval_results.sample_true_labels,
            eval_results.sample_pred_labels,
            idx2tag,
            output_path
        )
    
    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "num_labels": len(tag2idx),
                "pretrained_model_name": MODEL_NAME,
                "dropout_rate": args.dropout
            }
        },
        f'{output_path}/token_classifier_model.bin',
    )
    
    print(f"Model successfully saved at {output_path}/token_classifier_model.bin")


if __name__ == "__main__":
    main()