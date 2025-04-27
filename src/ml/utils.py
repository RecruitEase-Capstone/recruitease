import re
import json
import logging
import numpy as np
from tqdm import trange
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix


def convert_goldparse(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines = []
        with open(dataturks_JSON_FilePath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content'].replace("\n", " ")
            entities = []
            data_annotations = data['annotation']
            if data_annotations is not None:
                for annotation in data_annotations:
                    point = annotation['points'][0]
                    labels = annotation['label']
                    if not isinstance(labels, list):
                        labels = [labels]

                    for label in labels:
                        point_start = point['start']
                        point_end = point['end']
                        point_text = point['text']

                        lstrip_diff = len(point_text) - \
                            len(point_text.lstrip())
                        rstrip_diff = len(point_text) - \
                            len(point_text.rstrip())
                        if lstrip_diff != 0:
                            point_start = point_start + lstrip_diff
                        if rstrip_diff != 0:
                            point_end = point_end - rstrip_diff
                        entities.append((point_start, point_end + 1, label))
            training_data.append((text, {"entities": entities}))
        return training_data
    except Exception as e:
        logging.exception("Unable to process " +
                          dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None


def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    return cleaned_data


def get_label(offset, labels):
    if offset[0] == 0 and offset[1] == 0:
        return 'O'
    for label in labels:
        if offset[1] >= label[0] and offset[0] <= label[1]:
            return label[2]
    return 'O'


tags_vals = ["UNKNOWN", "O", "Name", "Degree", "Skills", "College Name", "Email Address",
             "Designation", "Companies worked at", "Graduation Year", "Years of Experience", "Location"]

tag2idx = {t: i for i, t in enumerate(tags_vals)}
idx2tag = {i: t for i, t in enumerate(tags_vals)}


def process_resume(data, tokenizer, tag2idx, max_len, is_test=False):
    tok = tokenizer.encode_plus(
        data[0], max_length=max_len, return_offsets_mapping=True)
    curr_sent = {'orig_labels': [], 'labels': []}

    padding_length = max_len - len(tok['input_ids'])

    if not is_test:
        labels = data[1]['entities']
        labels.reverse()
        for off in tok['offset_mapping']:
            label = get_label(off, labels)
            curr_sent['orig_labels'].append(label)
            curr_sent['labels'].append(tag2idx[label])
        curr_sent['labels'] = curr_sent['labels'] + ([0] * padding_length)

    curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + \
        ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + \
        ([0] * padding_length)
    return curr_sent

class ResumeDataset(Dataset):
    def __init__(self, resume, tokenizer, tag2idx, max_len, is_test=False):
        self.resume = resume
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.resume)

    def __getitem__(self, idx):
        data = process_resume(
            self.resume[idx], self.tokenizer, self.tag2idx, self.max_len, self.is_test)
        return {
            'input_ids': torch.tensor(data['input_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(data['token_type_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(data['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(data['labels'], dtype=torch.long),
            'orig_label': data['orig_labels']
        }

def get_hyperparameters(model, ff):

    # ff: full_finetuning
    if ff:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer]}]

    return optimizer_grouped_parameters

def get_special_tokens(tokenizer, tag2idx):
    vocab = tokenizer.get_vocab()
    pad_tok = vocab["[PAD]"]
    sep_tok = vocab["[SEP]"]
    cls_tok = vocab["[CLS]"]
    o_lab = tag2idx["O"]

    return pad_tok, sep_tok, cls_tok, o_lab

def annot_confusion_matrix(valid_tags, pred_tags):
    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn's `confusion_matrix`.
    """

    header = sorted(list(set(valid_tags + pred_tags)))

    matrix = confusion_matrix(valid_tags, pred_tags, labels=header)

    mat_formatted = [header[i] + "\t\t\t" +
                     str(row) for i, row in enumerate(matrix)]
    content = "\t" + " ".join(header) + "\n" + "\n".join(mat_formatted)

    return content

def flat_accuracy(valid_tags, pred_tags):
    return (np.array(valid_tags) == np.array(pred_tags)).mean()

def evaluate_model_with_report(model, dataloader, idx2tag, tag2idx, device):
    """
    Evaluates model and generates classification report
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        idx2tag: Dictionary to convert tag indices to tag strings
        tag2idx: Dictionary to convert tag strings to indices
        device: Device to run the model on (cuda/cpu)
    
    Returns:
        dict: Dictionary containing evaluation metrics (loss, accuracy) and classification report
    """
    model.eval()
    
    all_true_labels = []
    all_predicted_labels = []
    total_eval_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Process each sequence in the batch separately
            for i in range(input_ids.size(0)):
                # Get active tokens for this sequence (exclude [PAD], [CLS], [SEP])
                seq_mask = attention_mask[i] == 1
                seq_logits = logits[i][seq_mask]
                seq_labels = labels[i][seq_mask]
                
                # Get predictions
                _, seq_preds = torch.max(seq_logits, dim=1)
                
                # Convert to tag strings and collect
                true_labels = [idx2tag.get(label.item(), "O") for label in seq_labels]
                pred_labels = [idx2tag.get(pred.item(), "O") for pred in seq_preds]
                
                # Make sure lengths are identical
                min_len = min(len(true_labels), len(pred_labels))
                
                # Add as complete sequences (with equal length)
                if min_len > 0:
                    all_true_labels.append(true_labels[:min_len])
                    all_predicted_labels.append(pred_labels[:min_len])
                
                # Calculate token-level accuracy
                correct_predictions = (seq_preds == seq_labels).sum().item()
                total_correct += correct_predictions
                total_tokens += seq_mask.sum().item()
            
            total_eval_loss += loss.item()
    
    # Calculate metrics
    avg_loss = total_eval_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    # Generate classification report
    report = classification_report(all_true_labels, all_predicted_labels, digits=4)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "classification_report": report
    }

def train_and_val_model(model, tokenizer, optimizer, epochs, idx2tag, tag2idx, max_grad_norm, device, train_dataloader, val_dataloader, output_path='.'):
    """
    Melatih dan memvalidasi model dengan pengumpulan metrik untuk visualisasi
    
    Returns:
        tuple: (train_losses, val_losses, train_accuracies, val_accuracies)
    """
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Parameter early stopping
    best_val_loss = float('inf')
    patience = 5  # Jumlah epoch untuk menunggu sebelum early stopping
    counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_tokens = 0
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            model.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Calculate accuracy (token level)
            active_tokens = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, len(tag2idx))[active_tokens]
            active_labels = labels.view(-1)[active_tokens]
            
            _, predicted_tags = torch.max(active_logits, dim=1)
            correct_predictions = (predicted_tags == active_labels).sum().item()
            
            total_train_correct += correct_predictions
            total_train_tokens += active_tokens.sum().item()
            
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = total_train_correct / total_train_tokens if total_train_tokens > 0 else 0
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_tokens = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Calculate accuracy (token level)
                active_tokens = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, len(tag2idx))[active_tokens]
                active_labels = labels.view(-1)[active_tokens]
                
                _, predicted_tags = torch.max(active_logits, dim=1)
                correct_predictions = (predicted_tags == active_labels).sum().item()
                
                total_val_correct += correct_predictions
                total_val_tokens += active_tokens.sum().item()
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = total_val_correct / total_val_tokens if total_val_tokens > 0 else 0
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        print("------")
    
    return train_losses, val_losses, train_accuracies, val_accuracies