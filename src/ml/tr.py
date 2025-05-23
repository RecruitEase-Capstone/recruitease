import torch
import numpy as np
import json
import argparse
import re
import fitz
import os
from transformers import BertForTokenClassification, BertTokenizerFast, AutoTokenizer
from scipy.special import softmax

class ResumeNERPredictor:
    """
    A class to extract named entities from resume text using a BERT-based NER model.
    """
    
    def __init__(self, model_path=None, max_len=512, overlap=100):
        """
        Initialize the ResumeNERPredictor with a pre-trained model.
        
        Args:
            model_path (str): Path to the saved model file (.bin)
            max_len (int): Maximum sequence length for BERT input
            overlap (int): Number of tokens to overlap between chunks
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len
        self.overlap = overlap
        
        # Define entity mapping - sesuai dengan format output yang diinginkan
        self.entity_dict = {
            'Name': 'Name',
            'Degree': 'Degree', 
            'Skills': 'Skills',
            'College Name': 'College Name',
            'Email Address': 'Email Address',
            'Designation': 'Designation',
            'Companies worked at': 'Companies worked at',
            'Graduation Year': 'Graduation Year',
            'Years of Experience': 'Years of Experience',
            'Location': 'Location'
        }
        
        # Define tags - sesuai dengan kode asli
        self.tags_vals = ["UNKNOWN", "O", "Name", "Degree", "Skills", "College Name", "Email Address",
                         "Designation", "Companies worked at", "Graduation Year", "Years of Experience", "Location"]
        
        self.tag2idx = {t: i for i, t in enumerate(self.tags_vals)}
        self.idx2tag = {i: t for i, t in enumerate(self.tags_vals)}
        
        # Label yang tidak perlu ditampilkan
        self.restricted_labels = ['O', 'UNKNOWN']
        
        # Load tokenizer
        print("Loading tokenizer...")
        try:
            self.tokenizer = BertTokenizerFast('./vocab/vocab.txt', lowercase=True)
        except:
            print("Could not load custom vocab, using default tokenizer")
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        # Load model
        print(f"Loading model from {model_path if model_path else 'bert-base-uncased'}...")
        
        if model_path and os.path.exists(model_path):
            self.model = BertForTokenClassification.from_pretrained(
                'bert-base-uncased', 
                num_labels=len(self.tag2idx)
            )
            
            # Load state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully from {model_path}")
        else:
            self.model = BertForTokenClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=len(self.tag2idx)
            )
            print("No model path provided or file not found, using base model")
            
        self.model.to(self.device)
        self.model.eval()
    
    def pdf_to_text(self, pdf_path):
        """Convert PDF to text"""
        print("Processing PDF to extract text...")
        doc = fitz.open(pdf_path)
        
        text = ''
        for page in doc:
            text += page.get_text()
        
        # Remove new line dan normalize whitespace
        text = ' '.join(text.split('\n'))
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def split_text_by_sentences(self, text, max_sentence_length=400):
        """
        Membagi teks berdasarkan tanda titik (.) dengan mempertimbangkan panjang maksimal
        """
        # Split berdasarkan titik, tapi pertahankan titik
        sentences = re.split(r'(\. |\.$)', text)
        
        # Gabungkan kembali dengan titik
        processed_sentences = []
        current_sentence = ""
        
        for i, part in enumerate(sentences):
            if part in ['. ', '.$']:
                current_sentence += part
                if len(current_sentence.strip()) > 0:
                    processed_sentences.append(current_sentence.strip())
                    current_sentence = ""
            else:
                current_sentence += part
        
        # Tambahkan sisa teks jika ada
        if current_sentence.strip():
            processed_sentences.append(current_sentence.strip())
        
        # Jika masih ada kalimat yang terlalu panjang, split berdasarkan newline atau koma
        final_sentences = []
        for sentence in processed_sentences:
            if len(sentence) <= max_sentence_length:
                final_sentences.append(sentence)
            else:
                # Split berdasarkan newline atau koma jika terlalu panjang
                sub_parts = re.split(r'(\n|, )', sentence)
                current_part = ""
                
                for sub_part in sub_parts:
                    if len(current_part + sub_part) <= max_sentence_length:
                        current_part += sub_part
                    else:
                        if current_part.strip():
                            final_sentences.append(current_part.strip())
                        current_part = sub_part
                
                if current_part.strip():
                    final_sentences.append(current_part.strip())
        
        return [s for s in final_sentences if len(s.strip()) > 0]
    
    def tokenize_resume(self, resume_text):
        """
        Tokenize resume text menggunakan BERT tokenizer
        """
        encoding = self.tokenizer(
            resume_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'offset_mapping': encoding['offset_mapping'].squeeze()
        }
    
    def predict_single_sentence(self, sentence_text):
        """
        Prediksi entitas dari satu kalimat
        """
        data = self.tokenize_resume(sentence_text)
        
        # Prepare input untuk model
        input_ids = data['input_ids'].unsqueeze(0)
        input_mask = data['attention_mask'].unsqueeze(0)
        
        # Move ke device
        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                token_type_ids=None,
                attention_mask=input_mask,
            )
            logits = outputs.logits
        
        # Convert ke numpy dan dapatkan prediksi
        logits = logits.cpu().detach().numpy()
        label_ids = np.argmax(logits, axis=2)
        
        entities = []
        for label_id, offset in zip(label_ids[0], data['offset_mapping']):
            curr_id = self.idx2tag[label_id]
            curr_start = offset[0].item()
            curr_end = offset[1].item()
            
            # Skip restricted labels dan token [CLS], [SEP], [PAD]
            if curr_id not in self.restricted_labels and curr_start != curr_end:
                # Merge entitas yang bersebelahan dengan label yang sama
                if (len(entities) > 0 and 
                    entities[-1]['entity'] == curr_id and 
                    curr_start - entities[-1]['end'] in [0, 1]):
                    entities[-1]['end'] = curr_end
                else:
                    entities.append({
                        'entity': curr_id, 
                        'start': curr_start, 
                        'end': curr_end
                    })
        
        # Tambahkan teks untuk setiap entitas
        for ent in entities:
            ent['text'] = sentence_text[ent['start']:ent['end']].strip()
        
        return entities
    
    def merge_adjacent_entities(self, entities):
        """
        Merge entitas yang bersebelahan dengan label yang sama
        """
        if not entities:
            return entities
        
        # Sort berdasarkan posisi start
        entities.sort(key=lambda x: x['start'])
        
        merged = []
        current_entity = entities[0].copy()
        
        for entity in entities[1:]:
            # Jika entitas sama dan posisinya bersebelahan atau overlapping
            if (current_entity['entity'] == entity['entity'] and 
                entity['start'] <= current_entity['end'] + 5):  # tolerance 5 karakter
                # Extend current entity
                current_entity['end'] = max(current_entity['end'], entity['end'])
                current_entity['text'] = current_entity['text'] + " " + entity['text']
                current_entity['text'] = current_entity['text'].strip()
            else:
                merged.append(current_entity)
                current_entity = entity.copy()
        
        merged.append(current_entity)
        return merged
    
    def predict(self, text):
        """
        Extract named entities from input text by processing it in chunks
        
        Args:
            text (str): Resume text content
            
        Returns:
            dict: Dictionary with extracted entities (format sesuai kode referensi)
        """
        # Initialize result dictionary sesuai format yang diinginkan
        result_json = {
            'Name': [],
            'College Name': [],
            'Degree': [],
            'Graduation Year': [],
            'Years of Experience': [],
            'Companies worked at': [],
            'Designation': [],
            'Skills': [],
            'Location': [],
            'Email Address': []
        }
        
        print("Splitting resume into sentences...")
        sentences = self.split_text_by_sentences(text, max_sentence_length=self.max_len-100)
        print(f"Resume split into {len(sentences)} parts")
        
        all_entities = []
        current_position = 0
        
        for i, sentence in enumerate(sentences):
            print(f"Processing part {i+1}/{len(sentences)}: {sentence[:50]}...")
            
            # Prediksi untuk kalimat ini
            sentence_entities = self.predict_single_sentence(sentence)
            
            # Adjust posisi entitas ke posisi absolut dalam teks asli
            sentence_start_in_original = text.find(sentence, current_position)
            if sentence_start_in_original == -1:
                sentence_start_in_original = current_position
            
            for entity in sentence_entities:
                adjusted_entity = entity.copy()
                adjusted_entity['start'] = entity['start'] + sentence_start_in_original
                adjusted_entity['end'] = entity['end'] + sentence_start_in_original
                adjusted_entity['text'] = text[adjusted_entity['start']:adjusted_entity['end']].strip()
                all_entities.append(adjusted_entity)
            
            # Update posisi untuk kalimat berikutnya
            current_position = sentence_start_in_original + len(sentence)
        
        # Merge entitas yang bersebelahan dengan label yang sama
        merged_entities = self.merge_adjacent_entities(all_entities)
        
        # Konversi ke format output yang diinginkan
        for entity in merged_entities:
            entity_type = entity['entity']
            entity_text = entity['text']
            
            if entity_type in result_json and entity_text:
                # Remove duplicates
                if entity_text not in result_json[entity_type]:
                    result_json[entity_type].append(entity_text)
        
        # Post-process untuk membersihkan entitas
        for key in result_json:
            # Remove empty strings dan duplicates
            cleaned_entities = []
            seen = set()
            for entity in result_json[key]:
                cleaned_entity = entity.strip()
                if cleaned_entity and cleaned_entity not in seen:
                    seen.add(cleaned_entity)
                    cleaned_entities.append(cleaned_entity)
            result_json[key] = cleaned_entities
        
        return result_json
    
    def predict_from_pdf(self, pdf_path):
        """
        Extract named entities from a PDF file
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            dict: Dictionary with extracted entities
        """
        print(f"Extracting information from PDF: {pdf_path}")
        text = self.pdf_to_text(pdf_path)
        return self.predict(text)