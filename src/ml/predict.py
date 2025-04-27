import pandas as pd
import torch
import argparse
import json
import csv
import os
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertTokenizerFast
from tqdm import tqdm
import re

tags_vals = ["UNKNOWN", "O", "Name", "Degree", "Skills", "College Name", "Email Address",
             "Designation", "Companies worked at", "Graduation Year", "Years of Experience", "Location"]

tag2idx = {t: i for i, t in enumerate(tags_vals)}
idx2tag = {i: t for i, t in enumerate(tags_vals)}
restricted_labels = ["UNKNOWN", "O"]

def adjust_to_word_boundaries(text, start, end):
    """Adjust entity boundaries to align with word boundaries"""
    # Ensure we're within text bounds
    text_len = len(text)
    start = max(0, min(start, text_len - 1))
    end = max(0, min(end, text_len))
    
    # If start is in the middle of a word, move it to the beginning of the word
    while start > 0 and text[start-1].isalnum():
        start -= 1
    
    # If end is in the middle of a word, move it to the end of the word
    while end < text_len and text[end].isalnum():
        end += 1
    
    return start, end

def preprocess_text(text):
    """
    Preprocess text by:
    1. Replacing tab characters (\\t) with spaces
    2. Replacing newlines (\\n) with spaces
    3. Replacing Unicode escape sequences (\\u followed by 4 characters) with empty strings
    4. Replacing multiple spaces with a single space
    
    Args:
        text (str): The input text to preprocess
        
    Returns:
        str: The preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""  # Return empty string instead of None
    
    # Replace tabs with spaces
    processed_text = text.replace('\t', ' ')
    
    # Replace newlines with spaces
    processed_text = processed_text.replace('\n', ' ')
    
    # Replace Unicode escape sequences (\u followed by 4 characters)
    processed_text = re.sub(r'\\u[0-9a-fA-F]{4}', '', processed_text)
    
    # Replace multiple spaces with a single space
    processed_text = re.sub(r'\s+', ' ', processed_text)
    
    # Strip leading and trailing spaces
    processed_text = processed_text.strip()
    
    return processed_text

def tokenize_resume(text, tokenizer, max_len):
    """Tokenize resume text using the provided tokenizer"""
    tok = tokenizer.encode_plus(
        text, max_length=max_len, return_offsets_mapping=True, truncation=True)
    curr_sent = dict()
    padding_length = max_len - len(tok['input_ids'])
    curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + \
        ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + \
        ([0] * padding_length)
    final_data = {
        'input_ids': torch.tensor(curr_sent['input_ids'], dtype=torch.long),
        'token_type_ids': torch.tensor(curr_sent['token_type_ids'], dtype=torch.long),
        'attention_mask': torch.tensor(curr_sent['attention_mask'], dtype=torch.long),
        'offset_mapping': tok['offset_mapping']
    }
    return final_data

def predict_entities(text, model, tokenizer, device, max_length=500, threshold=0.5, debug=False):
    """Extract entities from text using the NER model"""
    text = preprocess_text(text)
    entities = []
    
    if not text or len(text.strip()) == 0:
        return entities
    
    # Ensure we don't exceed maximum token limit
    text = text[:max_length * 5]  # Rough estimate to avoid too many tokens
    
    # Use the provided tokenize_resume function
    tokenized = tokenize_resume(text, tokenizer, max_length)
    
    input_ids = tokenized['input_ids'].unsqueeze(0).to(device)
    attention_mask = tokenized['attention_mask'].unsqueeze(0).to(device)
    offset_mapping = tokenized['offset_mapping']
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=2)
    predictions = torch.argmax(logits, dim=2).cpu().numpy()[0]
    scores = probs.max(dim=2)[0].cpu().numpy()[0]
    
    # Process predictions to get entities
    current_entity = None
    
    for i, (pred, score, offset) in enumerate(zip(predictions, scores, offset_mapping)):
        # Skip padding tokens
        if attention_mask[0][i] == 0:
            continue
        
        # Get the predicted tag and score
        tag = idx2tag.get(pred, "UNKNOWN")
        
        # Skip if below threshold or in restricted labels
        if score < threshold or tag in restricted_labels:
            # End any current entity
            if current_entity:
                # Adjust boundaries to word boundaries
                start, end = adjust_to_word_boundaries(text, current_entity["start"], current_entity["end"])
                current_entity["start"] = start
                current_entity["end"] = end
                current_entity["text"] = text[start:end]
                entities.append(current_entity)
                current_entity = None
            continue
        
        # Extract the start and end position from offset mapping
        if isinstance(offset, tuple) and len(offset) == 2:
            start_pos, end_pos = offset
        else:
            continue
        
        # Skip special tokens
        if start_pos == 0 and end_pos == 0:
            continue
        
        # Start a new entity if one doesn't exist
        if current_entity is None:
            current_entity = {
                "type": tag,
                "start": start_pos,
                "end": end_pos,
                "text": text[start_pos:end_pos],
                "score": score.item()
            }
        else:
            # Check if it's the same entity type and continuous
            if current_entity["type"] == tag and start_pos <= current_entity["end"]:
                # Extend the entity
                current_entity["end"] = end_pos
                current_entity["text"] = text[current_entity["start"]:end_pos]
                # Update score (use average)
                current_entity["score"] = (current_entity["score"] + score.item()) / 2
            else:
                # End current entity and start a new one
                # Adjust boundaries to word boundaries
                start, end = adjust_to_word_boundaries(text, current_entity["start"], current_entity["end"])
                current_entity["start"] = start
                current_entity["end"] = end
                current_entity["text"] = text[start:end]
                entities.append(current_entity)
                current_entity = {
                    "type": tag,
                    "start": start_pos,
                    "end": end_pos,
                    "text": text[start_pos:end_pos],
                    "score": score.item()
                }
    
    # Add the last entity if it exists
    if current_entity:
        # Adjust boundaries to word boundaries
        start, end = adjust_to_word_boundaries(text, current_entity["start"], current_entity["end"])
        current_entity["start"] = start
        current_entity["end"] = end
        current_entity["text"] = text[start:end]
        entities.append(current_entity)
    
    return entities

def format_entities_dict(entities, debug=False):
    """Convert entities to a dictionary by entity type"""
    result = {}
    
    # Group entities by type
    for entity in entities:
        entity_type = entity["type"]
        
        if entity_type not in result:
            result[entity_type] = []
        
        if debug:
            # Include full entity details including score
            result[entity_type].append(entity)
        else:
            # Just include the text
            result[entity_type].append(entity["text"])
    
    return result

def print_entities(resume_id, entities_dict, debug=False):
    """Print extracted entities for a resume"""
    print(f"\nEntities for resume {resume_id}:")
    
    if not entities_dict:
        print("  No entities found")
        return
    
    for entity_type, values in entities_dict.items():
        print(f"  {entity_type}:")
        if debug and values and isinstance(values[0], dict):  # Check if values exist
            # Print with scores for debugging
            for v in values:
                print(f"    - {v['text']} (score: {v['score']:.4f})")
        else:
            # Regular printing
            for v in values:
                print(f"    - {v}")

def save_sample_output(text, entities, tokenizer_output, predictions_output, output_file):
    """Save sample output for debugging"""
    sample_data = {
        "text": text[:500] + "..." if len(text) > 500 else text,
        "entities": entities,
        "tokenizer_output": {
            "input_ids": tokenizer_output["input_ids"].tolist(),
            "attention_mask": tokenizer_output["attention_mask"].tolist()
        },
        "predictions": predictions_output
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample output saved to {output_file}")

def save_results(results, output_file, debug=False):
    """Save results to a CSV file"""
    # Prepare data for CSV
    csv_data = []
    
    for result in results:
        resume_id = result['ID']
        category = result['Category']
        entities = result['Entities']
        
        # Format entities as JSON strings for each entity type
        entity_columns = {}
        for entity_type, values in entities.items():
            if values:  # Check if values exist
                if debug and isinstance(values[0], dict):
                    # Format with scores for debugging
                    formatted_values = [f"{v['text']} ({v['score']:.4f})" for v in values]
                    entity_columns[entity_type] = "".join(formatted_values)
                else:
                    # Regular format without scores
                    entity_columns[entity_type] = "|".join(values)
            else:
                entity_columns[entity_type] = ""
        
        # Create row with all data
        row = {'ID': resume_id, 'Category': category}
        row.update(entity_columns)
        
        csv_data.append(row)

    print(f"Preparing to save CSV with {len(csv_data)} rows")
    if debug:
        print(f"First row keys: {csv_data[0].keys() if csv_data else 'No data'}")
    
    # Get all possible columns (entity types)
    all_columns = ['ID', 'Category']
    for result in results:
        for entity_type in result['Entities'].keys():
            if entity_type not in all_columns:
                all_columns.append(entity_type)
    
    print(f"All columns: {all_columns}")
    
    # Write to CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
            for row in csv_data:
                try:
                    writer.writerow(row)
                except Exception as e:
                    print(f"Error writing row: {e}")
                    print(f"Row: {row}")
                    # Try to fix encoding issues in row values
                    fixed_row = {k: str(v).encode('utf-8', 'ignore').decode('utf-8') if isinstance(v, str) else v 
                                for k, v in row.items()}
                    try:
                        writer.writerow(fixed_row)
                    except Exception as e2:
                        print(f"Still failed after fixing encoding: {e2}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        # Try with a different filename
        backup_output = output_file + ".backup.csv"
        print(f"Trying to save to backup file: {backup_output}")
        with open(backup_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
            for row in csv_data:
                writer.writerow({k: str(v).encode('utf-8', 'ignore').decode('utf-8') if isinstance(v, str) else v 
                              for k, v in row.items()})

def predict_entities_sliding_window(text, model, tokenizer, device, max_length=500, step=400, threshold=0.5, debug=False):
    """Extract entities from text using a sliding window approach for long texts"""
    text = preprocess_text(text)
    
    if not text or len(text.strip()) == 0:
        return []
    
    # Process the entire text in sliding windows
    text_length = len(text)
    all_windows_entities = []
    
    # If text is shorter than max_length, process it in one go
    if text_length <= max_length * 5:
        return predict_entities(text, model, tokenizer, device, max_length, threshold, debug)
    
    # Otherwise, process text in overlapping windows
    start_positions = list(range(0, text_length, step))
    
    if debug:
        print(f"Processing long text ({text_length} chars) in {len(start_positions)} windows")
    
    for start_pos in start_positions:
        end_pos = min(start_pos + max_length * 5, text_length)  # Take a large enough chunk for tokenization
        chunk_text = text[start_pos:end_pos]
        
        # Get window position in original text
        window_offset = start_pos
        
        # Process chunk
        chunk_entities = predict_entities(chunk_text, model, tokenizer, device, max_length, threshold, debug)
        
        # Adjust entity positions based on window offset
        for entity in chunk_entities:
            entity["start"] += window_offset
            entity["end"] += window_offset
            # Update entity text from the original text to handle boundary issues
            # Adjust to word boundaries
            start, end = adjust_to_word_boundaries(text, entity["start"], entity["end"])
            entity["start"] = start
            entity["end"] = end
            entity["text"] = text[start:end]
            all_windows_entities.append(entity)
    
    # Merge overlapping entities (prefer the one with higher score)
    all_windows_entities.sort(key=lambda e: (e["start"], -e["score"]))  # Sort by start pos, then by descending score
    merged_entities = []
    
    # Improved entity merging logic
    for entity in all_windows_entities:
        # Flag to check if current entity was merged with any existing entity
        merged = False
        
        # Check if this entity overlaps with any entity in merged_entities
        for i, existing_entity in enumerate(merged_entities):
            # Check for overlap
            if (entity["start"] <= existing_entity["end"] and 
                entity["end"] >= existing_entity["start"]):
                
                # If same entity type
                if entity["type"] == existing_entity["type"]:
                    # Calculate overlap extent
                    overlap_start = max(entity["start"], existing_entity["start"])
                    overlap_end = min(entity["end"], existing_entity["end"])
                    overlap_length = overlap_end - overlap_start
                    
                    # Significant overlap (more than 20% of either entity)
                    entity_length = entity["end"] - entity["start"]
                    existing_length = existing_entity["end"] - existing_entity["start"]
                    
                    if (overlap_length > 0.2 * entity_length or 
                        overlap_length > 0.2 * existing_length):
                        
                        # Merge based on score
                        if entity["score"] > existing_entity["score"]:
                            # Use the entity with the better score
                            # But keep the widest span
                            merged_start = min(entity["start"], existing_entity["start"])
                            merged_end = max(entity["end"], existing_entity["end"])
                            
                            # Adjust to word boundaries
                            merged_start, merged_end = adjust_to_word_boundaries(text, merged_start, merged_end)
                            
                            merged_entities[i] = {
                                "type": entity["type"],
                                "start": merged_start,
                                "end": merged_end,
                                "text": text[merged_start:merged_end],
                                "score": entity["score"]
                            }
                        else:
                            # Just extend the existing entity if needed
                            if entity["start"] < existing_entity["start"] or entity["end"] > existing_entity["end"]:
                                merged_start = min(entity["start"], existing_entity["start"])
                                merged_end = max(entity["end"], existing_entity["end"])
                                
                                # Adjust to word boundaries
                                merged_start, merged_end = adjust_to_word_boundaries(text, merged_start, merged_end)
                                
                                merged_entities[i] = {
                                    "type": existing_entity["type"],
                                    "start": merged_start,
                                    "end": merged_end,
                                    "text": text[merged_start:merged_end],
                                    "score": existing_entity["score"]
                                }
                        
                        merged = True
                        break
        
        # If the entity wasn't merged with any existing entity, add it
        if not merged:
            # Adjust once more to word boundaries before adding
            start, end = adjust_to_word_boundaries(text, entity["start"], entity["end"])
            entity["start"] = start
            entity["end"] = end
            entity["text"] = text[start:end]
            merged_entities.append(entity)
    
    return merged_entities

def process_csv(csv_file, model, tokenizer, device, output_file=None, batch_size=1, debug=False, threshold=0.5, sample_output=None, max_length=500, step=400, limit=None, use_sliding_window=True):
    """Process each row in the CSV file"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Apply limit if specified
        if limit and limit > 0:
            df = df.head(limit)
        
        # Check if required column exists
        if 'Resume_str' not in df.columns:
            print("Error: CSV file must contain 'Resume_str' column")
            return
        
        print(f"Processing {len(df)} resumes from CSV file...")
        
        # Prepare output data structure
        all_results = []
        
        # Save sample output for debugging if requested
        if sample_output and debug and not df.empty:
            # Get first row for sample
            sample_row = df.iloc[0]
            sample_text = sample_row['Resume_str']
            
            # Prepare tokenizer output for sample
            sample_tokenized = tokenize_resume(sample_text[:1000], tokenizer, max_length)
            
            # Get predictions for sample
            input_ids = sample_tokenized['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample_tokenized['attention_mask'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2).cpu().numpy()[0]
            
            # Format predictions
            prediction_tags = [idx2tag.get(pred, "UNKNOWN") for pred in predictions]
            
            # Extract entities
            if use_sliding_window:
                sample_entities = predict_entities_sliding_window(
                    sample_text[:1000], model, tokenizer, device, max_length, step, threshold, debug)
            else:
                sample_entities = predict_entities(
                    sample_text[:1000], model, tokenizer, device, max_length, threshold, debug)
            
            # Save sample output
            save_sample_output(
                sample_text[:1000],
                sample_entities,
                sample_tokenized,
                prediction_tags[:50],  # Just save first 50 predictions to keep file manageable
                sample_output
            )
        
        # Process each row with progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing resumes"):
            resume_id = row.get('ID', f"Row_{idx}")
            resume_text = row['Resume_str']
            category = row.get('Category', 'Unknown')
            
            if pd.isna(resume_text):
                resume_text = ""  # Ensure resume_text is never None/NA
            
            if debug and idx % 10 == 0:
                print(f"Processing resume {idx+1}/{len(df)} (ID: {resume_id})...")
                print(f"Resume length: {len(resume_text)} characters")
            
            # Get entities using sliding window if text is long
            if use_sliding_window:
                entities = predict_entities_sliding_window(
                    resume_text, model, tokenizer, device, max_length, step, threshold, debug)
            else:
                entities = predict_entities(
                    resume_text, model, tokenizer, device, max_length, threshold, debug)
            
            # Post-processing: remove duplicates and ensure word boundaries
            unique_entities = []
            seen_texts = set()
            
            for entity in entities:
                # Skip if we've seen this exact text before for this entity type
                entity_key = f"{entity['type']}:{entity['text']}"
                if entity_key in seen_texts:
                    continue
                
                seen_texts.add(entity_key)
                unique_entities.append(entity)
            
            # For debugging, print entity count
            if debug:
                print(f"Found {len(entities)} raw entities, {len(unique_entities)} unique entities for resume {resume_id}")
            
            # Format entities
            entities_dict = format_entities_dict(unique_entities, debug)
            
            # For debugging, print entity types found
            if debug:
                print(f"Entity types found: {list(entities_dict.keys())}")
            
            # Print entities for this resume if in debug mode
            if debug:
                print_entities(resume_id, entities_dict, debug)
            
            # Add to results
            result_row = {
                'ID': resume_id,
                'Category': category,
                'Entities': entities_dict
            }
            all_results.append(result_row)
            
            # Save incremental results if output file is specified
            if output_file and (idx + 1) % batch_size == 0:
                try:
                    save_results(all_results, output_file, debug)
                    print(f"Incremental results saved ({idx + 1}/{len(df)})")
                except Exception as e:
                    print(f"Error saving incremental results: {e}")
        
        # Save final results if output file is specified
        if output_file:
            try:
                save_results(all_results, output_file, debug)
                print(f"\nResults saved to {output_file}")
            except Exception as e:
                print(f"Error saving final results: {e}")
                # Try with a simplified backup approach
                backup_file = f"{output_file}.backup_simple.csv"
                print(f"Trying to save simplified results to {backup_file}")
                with open(backup_file, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Category', 'Raw_Entities'])
                    for result in all_results:
                        writer.writerow([result['ID'], result['Category'], str(result['Entities'])])
        
        return all_results
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract entities from resume text in CSV file')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file with Resume_str column')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model state file')
    parser.add_argument('--output_file', type=str, default='resume_entities.csv', help='Path to save the results CSV')
    parser.add_argument('--model_name', type=str, default='dslim/bert-base-NER', help='Base model name for tokenizer')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for saving incremental results')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional output')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for entity detection')
    parser.add_argument('--sample_output', type=str, default=None, help='Path to save a sample output for debugging')
    parser.add_argument('--max_length', type=int, default=500, help='Maximum token length for processing')
    parser.add_argument('--step', type=int, default=400, help='Step size for sliding window (default: 400)')
    parser.add_argument('--no_sliding_window', action='store_true', help='Disable sliding window for long texts')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of records to process')
    parser.add_argument('--verify_tags', action='store_true', help='Verify tag mappings with model config')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.isfile(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.")
        return
    
    if not os.path.isfile(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        return
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    try:
        tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Trying to load tokenizer with local files...")
        tokenizer = BertTokenizerFast.from_pretrained(args.model_name, local_files_only=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=len(idx2tag),
            ignore_mismatched_sizes=True
        )
        
        # Load trained model state
        state_dict = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully!")
        
        # Verify tag mappings if requested
        if args.verify_tags:
            try:
                # Try to extract label mappings from model config
                if hasattr(model.config, 'id2label') and model.config.id2label:
                    model_tags = model.config.id2label
                    print("Model's tag mappings:")
                    for idx, tag in model_tags.items():
                        print(f"  {idx}: {tag}")
                    
                    # Compare with our mappings
                    print("\nComparing with hard-coded mappings:")
                    for idx, tag in idx2tag.items():
                        model_tag = model_tags.get(str(idx), None)
                        if model_tag and model_tag != tag:
                            print(f"Warning: Mismatch at index {idx} - Hard-coded: '{tag}', Model: '{model_tag}'")
            except Exception as e:
                print(f"Error verifying tags: {e}")
        
        print(f"Tag mapping being used: {idx2tag}")
        
        # Process CSV file
        results = process_csv(
            args.csv_file, 
            model, 
            tokenizer, 
            device, 
            args.output_file, 
            args.batch_size, 
            args.debug, 
            args.threshold, 
            args.sample_output, 
            args.max_length,
            args.step,
            args.limit,
            not args.no_sliding_window  # Use sliding window by default
        )
        
        if results:
            print(f"Processing completed. Found entities for {len(results)} resumes.")
        else:
            print("Processing failed or no results found.")
    
    except Exception as e:
        print(f"Error loading or using model: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main()