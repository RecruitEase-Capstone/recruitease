import torch
import numpy as np
import json
import argparse
import re
import fitz
import os
from transformers import BertForTokenClassification, BertTokenizerFast, AutoTokenizer
from scipy.special import softmax
import string

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
        
        # Define entity mapping
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
        
        # Define tags
        self.tags_vals = ["UNKNOWN", "O", "Name", "Degree", "Skills", "College Name", "Email Address",
                         "Designation", "Companies worked at", "Graduation Year", "Years of Experience", "Location"]
        
        self.tag2idx = {t: i for i, t in enumerate(self.tags_vals)}
        self.idx2tag = {i: t for i, t in enumerate(self.tags_vals)}
        
        # Label yang tidak perlu ditampilkan
        self.restricted_labels = ['O', 'UNKNOWN']
        
        # Load tokenizer
        print("Loading tokenizer...")
        try:
            if os.path.exists('./vocab/vocab.txt'):
                self.tokenizer = BertTokenizerFast('./vocab/vocab.txt', lowercase=True)
            else:
                self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        except Exception as e:
            print(f"Could not load custom vocab ({e}), using default tokenizer")
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        # Load model
        print(f"Loading model from {model_path if model_path else 'bert-base-uncased'}...")
        
        if model_path and os.path.exists(model_path):
            self.model = BertForTokenClassification.from_pretrained(
                'bert-base-uncased', 
                num_labels=len(self.tag2idx)
            )
            
            # Load state dict
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using base model instead")
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
        try:
            doc = fitz.open(pdf_path)
            
            text = ''
            for page in doc:
                text += page.get_text()
            
            doc.close()
            
            # Remove new line dan normalize whitespace
            text = ' '.join(text.split('\n'))
            text = re.sub(r'\s+', ' ', text.strip())
            
            return text
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return ""
    
    def split_text_by_sentences(self, text, max_sentence_length=400):
        """
        Membagi teks berdasarkan tanda titik (.) dengan mempertimbangkan panjang maksimal
        """
        if not text:
            return []
            
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
        Prediksi entitas dari satu kalimat dengan perbaikan untuk menangani subword tokenization
        """
        if not sentence_text or not sentence_text.strip():
            return []
            
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
        
        # Rekonstruksi entitas dengan menangani subword tokenization
        entities = []
        current_entity = None
        
        for i, (label_id, offset) in enumerate(zip(label_ids[0], data['offset_mapping'])):
            if label_id >= len(self.idx2tag):
                continue
                
            curr_id = self.idx2tag[label_id]
            curr_start = offset[0].item()
            curr_end = offset[1].item()
            
            # Skip restricted labels dan token [CLS], [SEP], [PAD]
            if curr_id in self.restricted_labels or curr_start == curr_end:
                # Jika ada entitas yang sedang dibangun, selesaikan
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Jika ini adalah entitas baru atau berbeda dari yang sebelumnya
            if current_entity is None or current_entity['entity'] != curr_id:
                # Selesaikan entitas sebelumnya jika ada
                if current_entity is not None:
                    entities.append(current_entity)
                
                # Mulai entitas baru
                current_entity = {
                    'entity': curr_id,
                    'start': curr_start,
                    'end': curr_end
                }
            else:
                # Lanjutkan entitas yang sama (handle subword)
                # Periksa apakah token ini bersebelahan atau bagian dari kata yang sama
                gap = curr_start - current_entity['end']
                if gap <= 2:  # Toleransi untuk spasi atau karakter pemisah
                    current_entity['end'] = curr_end
                else:
                    # Gap terlalu besar, selesaikan entitas sebelumnya dan mulai yang baru
                    entities.append(current_entity)
                    current_entity = {
                        'entity': curr_id,
                        'start': curr_start,
                        'end': curr_end
                    }
        
        # Selesaikan entitas terakhir jika ada
        if current_entity is not None:
            entities.append(current_entity)
        
        # Tambahkan teks dan perluas batas untuk menangkap kata lengkap
        for ent in entities:
            # Perluas ke kiri untuk menangkap awal kata yang lengkap
            start_pos = ent['start']
            while start_pos > 0 and not sentence_text[start_pos-1].isspace():
                start_pos -= 1
            
            # Perluas ke kanan untuk menangkap akhir kata yang lengkap
            end_pos = ent['end']
            while end_pos < len(sentence_text) and not sentence_text[end_pos].isspace():
                end_pos += 1
            
            # Untuk nama, perluas lebih jauh untuk menangkap nama lengkap
            if ent['entity'] == 'Name':
                # Perluas ke kiri untuk menangkap nama depan
                while start_pos > 0 and sentence_text[start_pos-1:start_pos] not in ['.', ',', '\n', '\t']:
                    if sentence_text[start_pos-1].isspace():
                        # Periksa apakah kata sebelumnya adalah bagian dari nama
                        prev_word_start = start_pos - 1
                        while prev_word_start > 0 and sentence_text[prev_word_start-1].isspace():
                            prev_word_start -= 1
                        while prev_word_start > 0 and not sentence_text[prev_word_start-1].isspace():
                            prev_word_start -= 1
                        
                        prev_word = sentence_text[prev_word_start:start_pos-1].strip()
                        # Jika kata sebelumnya adalah nama (huruf kapital dan bukan kata umum)
                        if (prev_word and prev_word[0].isupper() and 
                            prev_word.lower() not in ['mr', 'mrs', 'ms', 'dr', 'prof', 'the', 'and']):
                            start_pos = prev_word_start
                        else:
                            break
                    else:
                        start_pos -= 1
                
                # Perluas ke kanan untuk menangkap nama belakang
                while end_pos < len(sentence_text) and sentence_text[end_pos:end_pos+1] not in ['.', ',', '\n', '\t']:
                    if sentence_text[end_pos].isspace():
                        # Periksa apakah kata berikutnya adalah bagian dari nama
                        next_word_end = end_pos + 1
                        while next_word_end < len(sentence_text) and sentence_text[next_word_end].isspace():
                            next_word_end += 1
                        next_word_start = next_word_end
                        while next_word_end < len(sentence_text) and not sentence_text[next_word_end].isspace():
                            next_word_end += 1
                        
                        next_word = sentence_text[next_word_start:next_word_end].strip()
                        # Jika kata berikutnya adalah nama (huruf kapital dan bukan kata umum)
                        if (next_word and next_word[0].isupper() and 
                            next_word.lower() not in ['born', 'from', 'in', 'at', 'the', 'and', 'email', 'phone']):
                            end_pos = next_word_end
                        else:
                            break
                    else:
                        end_pos += 1
            
            # Update posisi dan teks entitas
            ent['start'] = start_pos
            ent['end'] = end_pos
            ent['text'] = sentence_text[start_pos:end_pos].strip()
        
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
                entity['start'] <= current_entity['end'] + 10):  # tolerance 10 karakter
                # Extend current entity
                current_entity['end'] = max(current_entity['end'], entity['end'])
                # Hindari duplikasi text saat menggabungkan
                if not current_entity['text'].endswith(entity['text']):
                    current_entity['text'] = (current_entity['text'] + " " + entity['text']).strip()
            else:
                merged.append(current_entity)
                current_entity = entity.copy()
        
        merged.append(current_entity)
        return merged

    def is_valid_entity(self, text, entity_type):
        """
        Validate if the extracted entity is a valid whole word/phrase
        
        Args:
            text (str): Entity text to validate
            entity_type (str): Type of entity
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        # Remove entities that are too short (less than 2 characters), except for initials in names
        if len(text) < 2 and entity_type != 'Name':
            return False
        elif entity_type == 'Name' and len(text) < 1:
            return False
        
        # Remove entities that are only punctuation or symbols
        if all(c in string.punctuation for c in text):
            return False
        
        # Remove entities that are only numbers (unless it's graduation year)
        if text.isdigit() and entity_type != 'Graduation Year':
            return False
        
        # Remove entities with excessive punctuation
        punct_ratio = sum(1 for c in text if c in string.punctuation) / len(text)
        if punct_ratio > 0.5:  # More than 50% punctuation
            return False
        
        # Specific validation for each entity type
        if entity_type == 'Name':
            # Names should contain alphabetic characters
            if not any(c.isalpha() for c in text):
                return False
            
            # Remove common non-name words and fragments
            non_name_words = {
                'resume', 'cv', 'curriculum', 'vitae', 'profile', 'summary', 'objective',
                'email', 'phone', 'address', 'contact', 'experience', 'education',
                'skills', 'work', 'employment', 'career', 'professional', 'personal',
                'information', 'details', 'background', 'qualification', 'achievement'
            }
            if text.lower() in non_name_words:
                return False
            
            # Remove single characters unless they are initials (uppercase)
            if len(text) == 1 and not text.isupper():
                return False
            
            # Remove fragments that are clearly not names
            short_fragments = {'ar', 'al', 'bin', 'el', 'la', 'le', 'de', 'da', 'di'}
            if len(text) <= 3 and text.lower() in short_fragments:
                return False
            
            # Validate that it looks like a real name (contains vowels for longer names)
            if len(text) >= 4:
                vowels = set('aeiouAEIOU')
                if not any(c in vowels for c in text):
                    return False
        
        elif entity_type == 'Graduation Year':
            # Should be a 4-digit year between 1950-2030
            if not re.match(r'^\d{4}$', text):
                return False
            year = int(text)
            if year < 1950 or year > 2030:
                return False
        
        elif entity_type == 'Years of Experience':
            # Should contain numbers and experience-related keywords
            if not any(c.isdigit() for c in text):
                return False
        
        elif entity_type == 'Skills':
            # Skills should not be too generic
            generic_skills = {'skill', 'skills', 'technical', 'soft', 'hard'}
            if text.lower() in generic_skills:
                return False
        
        elif entity_type == 'Location':
            # Should contain alphabetic characters
            if not any(c.isalpha() for c in text):
                return False
        
        elif entity_type in ['College Name', 'Companies worked at', 'Degree', 'Designation']:
            # Should contain alphabetic characters
            if not any(c.isalpha() for c in text):
                return False
            # Remove single character entities
            if len(text.replace(' ', '')) < 2:
                return False
        
        return True

    def clean_entity_text(self, text, entity_type):
        """
        Clean and normalize entity text
        
        Args:
            text (str): Entity text to clean
            entity_type (str): Type of entity
            
        Returns:
            str: Cleaned entity text
        """
        if not text:
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing punctuation (except for emails)
        if entity_type != 'Email Address':
            text = text.strip(string.punctuation + ' ')
        
        # Specific cleaning for each entity type
        if entity_type == 'Name':
            # Capitalize first letters
            text = ' '.join(word.capitalize() for word in text.split())
        
        elif entity_type == 'Email Address':
            # Convert to lowercase
            text = text.lower()
        
        elif entity_type in ['College Name', 'Companies worked at']:
            # Basic title case
            text = text.title()
        
        elif entity_type == 'Skills':
            # Remove common prefixes/suffixes
            prefixes = ['skill in', 'experience in', 'knowledge of']
            for prefix in prefixes:
                if text.lower().startswith(prefix):
                    text = text[len(prefix):].strip()
            
            # Capitalize appropriately
            text = text.title()
        
        return text

    def deduplicate_entities(self, entities_list):
        """
        Remove duplicates and similar entities from a list, with special handling for names
        
        Args:
            entities_list (list): List of entity strings
            
        Returns:
            list: Deduplicated list
        """
        if not entities_list:
            return []
        
        # Convert to lowercase for comparison
        seen = set()
        unique_entities = []
        
        # Special handling for names - merge fragments
        if ('Name' in str(entities_list)):  # Check if this might be processing names
            # Try to reconstruct full names from fragments
            name_parts = []
            for entity in entities_list:
                entity = entity.strip()
                if entity:
                    name_parts.extend(entity.split())
            
            # If we have multiple parts, try to reconstruct the full name
            if len(name_parts) > 1:
                # Look for patterns like "Muhammad Bin Djafar Almasyhur"
                reconstructed = ' '.join(name_parts)
                # Remove duplicates in the reconstructed name
                unique_parts = []
                for part in name_parts:
                    if part not in unique_parts:
                        unique_parts.append(part)
                
                if len(unique_parts) > 1:
                    return [' '.join(unique_parts)]
        
        for entity in entities_list:
            entity_lower = entity.lower().strip()
            
            # Skip if we've seen this exact entity
            if entity_lower in seen:
                continue
                
            # Check for substring matches (avoid partial duplicates)
            is_substring = False
            for existing in list(seen):  # Create a copy to avoid modification during iteration
                if entity_lower in existing or existing in entity_lower:
                    # Keep the longer version
                    if len(entity_lower) > len(existing):
                        # Remove the shorter version and add the longer one
                        unique_entities = [e for e in unique_entities if e.lower().strip() != existing]
                        seen.discard(existing)
                        break
                    else:
                        is_substring = True
                        break
            
            if not is_substring:
                seen.add(entity_lower)
                unique_entities.append(entity)
        
        return unique_entities

    def post_process_entities(self, result_json):
        """
        Post-process the extracted entities to improve quality
        
        Args:
            result_json (dict): Raw extracted entities
            
        Returns:
            dict: Post-processed entities
        """
        processed_result = {}
        
        for entity_type, entities in result_json.items():
            # First, clean and validate each entity
            cleaned_entities = []
            for entity in entities:
                cleaned = self.clean_entity_text(entity, entity_type)
                if cleaned and self.is_valid_entity(cleaned, entity_type):
                    cleaned_entities.append(cleaned)
            
            # Remove duplicates and similar entities
            processed_result[entity_type] = self.deduplicate_entities(cleaned_entities)
        
        return processed_result

    def predict(self, text):
        """
        Extract named entities from input text by processing it in chunks
        
        Args:
            text (str): Resume text content
            
        Returns:
            dict: Dictionary with extracted entities
        """
        # Initialize result dictionary
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
        
        if not text or not text.strip():
            print("Warning: Empty text provided")
            return result_json
        
        print("Splitting resume into sentences...")
        sentences = self.split_text_by_sentences(text, max_sentence_length=self.max_len-100)
        print(f"Resume split into {len(sentences)} parts")
        
        all_entities = []
        
        for i, sentence in enumerate(sentences):
            print(f"Processing part {i+1}/{len(sentences)}: {sentence[:50]}...")
            
            # Prediksi untuk kalimat ini
            sentence_entities = self.predict_single_sentence(sentence)
            
            # Langsung gunakan posisi relatif dalam kalimat
            for entity in sentence_entities:
                entity['original_text'] = sentence  # Simpan kalimat asli untuk reference
                all_entities.append(entity)
        
        # Merge entitas yang bersebelahan dengan label yang sama
        print("Merging adjacent entities...")
        merged_entities = self.merge_adjacent_entities(all_entities)
        
        # Konversi ke format output yang diinginkan
        print("Converting to output format...")
        for entity in merged_entities:
            entity_type = entity['entity']
            entity_text = entity['text']
            
            if entity_type in result_json and entity_text:
                result_json[entity_type].append(entity_text)
        
        # Post-process untuk membersihkan dan deduplikasi entitas
        print("Post-processing entities...")
        final_result = self.post_process_entities(result_json)
        
        # Urutkan hasil untuk konsistensi
        for key in final_result:
            if final_result[key]:
                final_result[key].sort()
        
        print("Entity extraction completed!")
        print(f"Extracted entities summary:")
        for key, values in final_result.items():
            if values:
                print(f"  {key}: {len(values)} items")
        
        return final_result
    
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
        if not text:
            print("Warning: No text extracted from PDF")
            return {key: [] for key in self.entity_dict.keys()}
        return self.predict(text)

def export_results(results, output_path, format='json'):
    """
    Export extraction results to different formats
    
    Args:
        results (dict): Extraction results
        output_path (str): Output file path
        format (str): Output format ('json', 'txt', 'csv')
    """
    try:
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for entity_type, entities in results.items():
                    if entities:
                        f.write(f"{entity_type}:\n")
                        for entity in entities:
                            f.write(f"  - {entity}\n")
                        f.write("\n")
        
        elif format == 'csv':
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Entity Type', 'Value'])
                for entity_type, entities in results.items():
                    for entity in entities:
                        writer.writerow([entity_type, entity])
        
        print(f"Results exported to {output_path}")
    except Exception as e:
        print(f"Error exporting results: {e}")

# if __name__ == "__main__":
#     predictor = ResumeNERPredictor('results/model-state.bin')
#     results = predictor.predict_from_pdf('notebooks/cv_izra.pdf')
#     print(results)
        