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
    # PERUBAHAN: Tambahkan truncation=True dan padding='max_length'
    tok = tokenizer.encode_plus(
        data[0], 
        max_length=max_len, 
        truncation=True,          # ← TAMBAHKAN INI
        padding='max_length',     # ← TAMBAHKAN INI 
        return_offsets_mapping=True
    )
    curr_sent = {'orig_labels': [], 'labels': []}

    # HAPUS manual padding karena sudah ada padding='max_length'
    # padding_length = max_len - len(tok['input_ids'])  # ← HAPUS BARIS INI

    if not is_test:
        labels = data[1]['entities']
        labels.reverse()
        for off in tok['offset_mapping']:
            label = get_label(off, labels)
            curr_sent['orig_labels'].append(label)
            curr_sent['labels'].append(tag2idx[label])
        
        # PERUBAHAN: Pad labels ke max_length jika diperlukan
        while len(curr_sent['labels']) < max_len:
            curr_sent['labels'].append(0)
        
        # Truncate labels jika lebih panjang dari max_len
        curr_sent['labels'] = curr_sent['labels'][:max_len]

    # PERUBAHAN: Tidak perlu manual padding lagi
    curr_sent['input_ids'] = tok['input_ids']
    curr_sent['token_type_ids'] = tok['token_type_ids']
    curr_sent['attention_mask'] = tok['attention_mask']
    
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


def train_and_val_model(
    model,
    tokenizer,
    optimizer,
    epochs,
    idx2tag,
    tag2idx,
    max_grad_norm,
    device,
    train_dataloader,
    valid_dataloader
):

    pad_tok, sep_tok, cls_tok, o_lab = get_special_tokens(tokenizer, tag2idx)

    epoch = 0
    for _ in trange(epochs, desc="Epoch"):
        epoch += 1

        # Training loop
        print("Starting training loop.")
        model.train()
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []

        for step, batch in enumerate(train_dataloader):
            # Add batch to gpu

            # batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            b_input_ids, b_input_mask, b_labels = b_input_ids.to(
                device), b_input_mask.to(device), b_labels.to(device)

            # Forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            loss, tr_logits = outputs[:2]

            # Backward pass
            loss.backward()

            # Compute train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            # Subset out unwanted predictions on CLS/PAD/SEP tokens
            preds_mask = (
                (b_input_ids != cls_tok)
                & (b_input_ids != pad_tok)
                & (b_input_ids != sep_tok)
            )

            tr_logits = tr_logits.cpu().detach().numpy()
            tr_label_ids = torch.masked_select(b_labels, (preds_mask == 1))
            preds_mask = preds_mask.cpu().detach().numpy()
            tr_batch_preds = np.argmax(tr_logits[preds_mask.squeeze()], axis=1)
            tr_batch_labels = tr_label_ids.to("cpu").numpy()
            tr_preds.extend(tr_batch_preds)
            tr_labels.extend(tr_batch_labels)

            # Compute training accuracy
            tmp_tr_accuracy = flat_accuracy(tr_batch_labels, tr_batch_preds)
            tr_accuracy += tmp_tr_accuracy

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_grad_norm
            )

            # Update parameters
            optimizer.step()
            model.zero_grad()

        tr_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps

        # Print training loss and accuracy per epoch
        print(f"Train loss: {tr_loss}")
        print(f"Train accuracy: {tr_accuracy}")

        """
        Validation loop
        """
        print("Starting validation loop.")

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        
        # PERBAIKAN: Inisialisasi list untuk format seqeval yang benar
        all_valid_tags = []  # List of lists untuk setiap sequence
        all_pred_tags = []   # List of lists untuk setiap sequence

        for batch in valid_dataloader:

            b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            b_input_ids, b_input_mask, b_labels = b_input_ids.to(
                device), b_input_mask.to(device), b_labels.to(device)

            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                tmp_eval_loss, logits = outputs[:2]

            # PERBAIKAN: Convert ke format seqeval yang benar
            predictions_tensor = torch.argmax(logits, dim=-1)
            
            # Convert untuk setiap sequence dalam batch
            batch_valid_tags = []
            batch_pred_tags = []
            
            for i in range(predictions_tensor.shape[0]):
                # Ambil sequence length berdasarkan attention mask
                seq_len = b_input_mask[i].sum().item()
                
                # Skip [CLS] token (index 0) dan ambil sampai [SEP] token
                valid_indices = range(1, seq_len - 1)
                
                seq_valid_tags = []
                seq_pred_tags = []
                
                for j in valid_indices:
                    # Skip padding tokens (label = -100)
                    if b_labels[i][j] != -100:
                        # Skip special tokens
                        token_id = b_input_ids[i][j].item()
                        if token_id not in [cls_tok, pad_tok, sep_tok]:
                            seq_valid_tags.append(idx2tag[b_labels[i][j].item()])
                            seq_pred_tags.append(idx2tag[predictions_tensor[i][j].item()])
                
                # Hanya tambahkan jika sequence tidak kosong
                if seq_valid_tags and seq_pred_tags:
                    batch_valid_tags.append(seq_valid_tags)
                    batch_pred_tags.append(seq_pred_tags)
            
            # Extend ke list utama
            all_valid_tags.extend(batch_valid_tags)
            all_pred_tags.extend(batch_pred_tags)

            # Hitung accuracy untuk kompatibilitas dengan kode lama
            preds_mask = (
                (b_input_ids != cls_tok)
                & (b_input_ids != pad_tok)
                & (b_input_ids != sep_tok)
            )

            logits_numpy = logits.cpu().detach().numpy()
            label_ids = torch.masked_select(b_labels, (preds_mask == 1))
            preds_mask_numpy = preds_mask.cpu().detach().numpy()
            val_batch_preds = np.argmax(logits_numpy[preds_mask_numpy.squeeze()], axis=1)
            val_batch_labels = label_ids.to("cpu").numpy()

            tmp_eval_accuracy = flat_accuracy(val_batch_labels, val_batch_preds)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        # PERBAIKAN: Debug format data sebelum classification report
        print(f"Debug - Type of all_valid_tags: {type(all_valid_tags)}")
        print(f"Debug - Length of all_valid_tags: {len(all_valid_tags)}")
        if all_valid_tags:
            print(f"Debug - Type of all_valid_tags[0]: {type(all_valid_tags[0])}")
            print(f"Debug - Sample all_valid_tags[0]: {all_valid_tags[0][:5] if len(all_valid_tags[0]) > 5 else all_valid_tags[0]}")
        
        print(f"Debug - Type of all_pred_tags: {type(all_pred_tags)}")
        print(f"Debug - Length of all_pred_tags: {len(all_pred_tags)}")
        if all_pred_tags:
            print(f"Debug - Type of all_pred_tags[0]: {type(all_pred_tags[0])}")
            print(f"Debug - Sample all_pred_tags[0]: {all_pred_tags[0][:5] if len(all_pred_tags[0]) > 5 else all_pred_tags[0]}")

        # Evaluate loss, acc, conf. matrix, and class. report on devset
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps

        # PERBAIKAN: Generate classification report dengan format yang benar
        try:
            if all_valid_tags and all_pred_tags and len(all_valid_tags) == len(all_pred_tags):
                # Clean data untuk memastikan tidak ada nested lists atau masalah lain
                cleaned_valid_tags = []
                cleaned_pred_tags = []
                
                for i in range(len(all_valid_tags)):
                    if isinstance(all_valid_tags[i], list) and isinstance(all_pred_tags[i], list):
                        # Pastikan semua elemen adalah string
                        valid_seq = [str(tag) for tag in all_valid_tags[i] if tag is not None]
                        pred_seq = [str(tag) for tag in all_pred_tags[i] if tag is not None]
                        
                        # Hanya tambahkan jika kedua sequence memiliki panjang yang sama dan tidak kosong
                        if len(valid_seq) == len(pred_seq) and len(valid_seq) > 0:
                            cleaned_valid_tags.append(valid_seq)
                            cleaned_pred_tags.append(pred_seq)
                
                print(f"Cleaned sequences: {len(cleaned_valid_tags)}")
                
                if cleaned_valid_tags and cleaned_pred_tags:
                    # Coba dengan sklearn classification_report sebagai alternatif
                    try:
                        from sklearn.metrics import classification_report as sklearn_report
                        from sklearn.metrics import confusion_matrix
                        
                        # Flatten untuk sklearn
                        flat_valid = [tag for seq in cleaned_valid_tags for tag in seq]
                        flat_pred = [tag for seq in cleaned_pred_tags for tag in seq]
                        
                        sklearn_cl_report = sklearn_report(flat_valid, flat_pred, zero_division=0)
                        
                        print(f"Validation loss: {eval_loss}")
                        print(f"Validation Accuracy: {eval_accuracy}")
                        print(f"Classification Report (sklearn):\n{sklearn_cl_report}")
                        
                        # Confusion matrix
                        unique_labels = sorted(list(set(flat_valid + flat_pred)))
                        conf_matrix = confusion_matrix(flat_valid, flat_pred, labels=unique_labels)
                        print(f"Confusion Matrix Labels: {unique_labels}")
                        print(f"Confusion Matrix:\n{conf_matrix}")
                        
                    except ImportError:
                        # Jika sklearn tidak tersedia, gunakan seqeval
                        cl_report = classification_report(cleaned_valid_tags, cleaned_pred_tags)
                        
                        print(f"Validation loss: {eval_loss}")
                        print(f"Validation Accuracy: {eval_accuracy}")
                        print(f"Classification Report (seqeval):\n{cl_report}")
                        
                        # Coba confusion matrix
                        try:
                            conf_mat = annot_confusion_matrix(cleaned_valid_tags, cleaned_pred_tags)
                            print(f"Confusion Matrix:\n{conf_mat}")
                        except:
                            print("Could not generate confusion matrix")
                            
                else:
                    print(f"Validation loss: {eval_loss}")
                    print(f"Validation Accuracy: {eval_accuracy}")
                    print("Warning: No valid sequences after cleaning")
                    
            else:
                print(f"Validation loss: {eval_loss}")
                print(f"Validation Accuracy: {eval_accuracy}")
                print("Warning: Cannot generate classification report due to data format issues")
                print(f"Valid tags length: {len(all_valid_tags) if all_valid_tags else 0}")
                print(f"Pred tags length: {len(all_pred_tags) if all_pred_tags else 0}")
                
        except Exception as e:
            print(f"Error generating classification report: {e}")
            print(f"Validation loss: {eval_loss}")
            print(f"Validation Accuracy: {eval_accuracy}")
            
            # Alternative: Manual metrics calculation
            try:
                # Flatten all data
                flat_valid_tags = [tag for seq in all_valid_tags for tag in seq if seq and tag]
                flat_pred_tags = [tag for seq in all_pred_tags for tag in seq if seq and tag]
                
                if len(flat_valid_tags) == len(flat_pred_tags):
                    # Calculate basic metrics manually
                    correct = sum(1 for v, p in zip(flat_valid_tags, flat_pred_tags) if v == p)
                    total = len(flat_valid_tags)
                    token_accuracy = correct / total if total > 0 else 0
                    
                    # Count by class
                    unique_tags = set(flat_valid_tags + flat_pred_tags)
                    tag_stats = {}
                    
                    for tag in unique_tags:
                        true_pos = sum(1 for v, p in zip(flat_valid_tags, flat_pred_tags) if v == tag and p == tag)
                        false_pos = sum(1 for v, p in zip(flat_valid_tags, flat_pred_tags) if v != tag and p == tag)
                        false_neg = sum(1 for v, p in zip(flat_valid_tags, flat_pred_tags) if v == tag and p != tag)
                        
                        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        tag_stats[tag] = {
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'support': sum(1 for v in flat_valid_tags if v == tag)
                        }
                    
                    print("\nManual Classification Report:")
                    print(f"Token Accuracy: {token_accuracy:.4f}")
                    print(f"{'Tag':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
                    print("-" * 70)
                    
                    for tag, stats in sorted(tag_stats.items()):
                        print(f"{tag:<20} {stats['precision']:<10.4f} {stats['recall']:<10.4f} {stats['f1']:<10.4f} {stats['support']:<10}")
                    
                    print(f"\nUnique predicted tags: {set(flat_pred_tags)}")
                    print(f"Unique true tags: {set(flat_valid_tags)}")
                else:
                    print(f"Length mismatch: valid={len(flat_valid_tags)}, pred={len(flat_pred_tags)}")
                    
            except Exception as manual_error:
                print(f"Manual calculation also failed: {manual_error}")
                print("Only showing basic validation metrics")
                print(f"Unique predicted tags: {set([tag for seq in all_pred_tags for tag in seq if seq and tag])}")
                print(f"Unique true tags: {set([tag for seq in all_valid_tags for tag in seq if seq and tag])}")

# Fungsi helper untuk debugging
def debug_seqeval_format(valid_tags, pred_tags):
    """
    Debug function untuk memeriksa format data seqeval
    """
    print("=== DEBUG SEQEVAL FORMAT ===")
    print(f"valid_tags type: {type(valid_tags)}")
    print(f"pred_tags type: {type(pred_tags)}")
    print(f"valid_tags length: {len(valid_tags)}")
    print(f"pred_tags length: {len(pred_tags)}")
    
    if valid_tags and len(valid_tags) > 0:
        print(f"valid_tags[0] type: {type(valid_tags[0])}")
        print(f"valid_tags[0]: {valid_tags[0]}")
        
    if pred_tags and len(pred_tags) > 0:
        print(f"pred_tags[0] type: {type(pred_tags[0])}")
        print(f"pred_tags[0]: {pred_tags[0]}")
    
    print("=== END DEBUG ===")