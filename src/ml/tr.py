import torch
import numpy as np
import fitz  # PyMuPDF
import os
from transformers import AutoTokenizer, BertForTokenClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.special import softmax
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


class ResumeNERPredictor:
    """
    A class to extract named entities from resume text using a BERT-based NER model.
    """
    
    def __init__(self, model_path=None, max_len=512):
        """
        Initialize the ResumeNERPredictor with a pre-trained model.
        
        Args:
            model_path (str): Path to the saved model file (.bin)
            max_len (int): Maximum sequence length for BERT input
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len
        
        # Define entity mapping
        self.entity_dict = {
            'NAME': 'Name',
            'CLG': 'College Name',
            'DEG': 'Degree',
            'GRADYEAR': 'Graduation Year',
            'YOE': 'Years of Experience',
            'COMPANY': 'Companies worked at',
            'DESIG': 'Designation',
            'SKILLS': 'Skills',
            'LOC': 'Location',
            'EMAIL': 'Email Address'
        }
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER', do_lower_case=True)
        
        # Define tags
        tag_vals = set(['X', '[CLS]', '[SEP]', 'O', 
                       'B-NAME', 'I-NAME', 'L-NAME', 'U-NAME',
                       'B-CLG', 'I-CLG', 'L-CLG', 'U-CLG',
                       'B-DEG', 'I-DEG', 'L-DEG', 'U-DEG',
                       'B-GRADYEAR', 'I-GRADYEAR', 'L-GRADYEAR', 'U-GRADYEAR',
                       'B-YOE', 'I-YOE', 'L-YOE', 'U-YOE',
                       'B-COMPANY', 'I-COMPANY', 'L-COMPANY', 'U-COMPANY',
                       'B-DESIG', 'I-DESIG', 'L-DESIG', 'U-DESIG',
                       'B-SKILLS', 'I-SKILLS', 'L-SKILLS', 'U-SKILLS',
                       'B-LOC', 'I-LOC', 'L-LOC', 'U-LOC',
                       'B-EMAIL', 'I-EMAIL', 'L-EMAIL', 'U-EMAIL'])
        
        self.tag2idx = {t: i for i, t in enumerate(tag_vals)}
        self.idx2tag = {i: t for t, i in self.tag2idx.items()}
        
        # Load model
        print(f"Loading model from {model_path if model_path else 'dslim/bert-base-NER'}...")
        self.model = BertForTokenClassification.from_pretrained(
            'dslim/bert-base-NER' if model_path is None else 'bert-base-uncased',
            num_labels=len(self.tag2idx),
            id2label=self.idx2tag,
            label2id=self.tag2idx
        )
        
        # Load weights if model path is provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded successfully from {model_path}")
        else:
            print("No model path provided or file not found, using base model")
            
        self.model.to(self.device)
        self.model.eval()
        
    def get_wordnet_pos(self, word):
        """Get wordnet POS tag from word"""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    def preprocess_text(self, text):
        """Preprocess text with tokenization, stopword removal and lemmatization"""
        # Tokenization
        tokenized_text = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_text = []
        for token in tokenized_text:
            if token not in stop_words:
                filtered_text.append(token)

        # POS and lemmatize
        lemmatizer = WordNetLemmatizer()
        lemmatized_results = [lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) for token in filtered_text]
        return ' '.join(lemmatized_results)
    
    def pdf_to_text(self, pdf_path, preprocessing=False):
        """Convert PDF to text"""
        # Open pdf file
        doc = fitz.open(pdf_path)

        # Convert pdf to text
        text = ''
        for page in doc:
            text += page.get_text()

        # Remove new line
        text = ' '.join(text.split('\n'))

        if preprocessing:
            return self.preprocess_text(text)
        else:
            return text
    
    def reconstruct_entity_text(self, tokens):
        """Reconstruct entity text from tokens, handling BERT's subword tokenization"""
        text = ""
        for token in tokens:
            if token.startswith('##'):
                text += token[2:]
            elif token.startswith('#'):
                text += token[1:]
            else:
                if text:
                    text += ' ' + token
                else:
                    text += token
        return text
    
    def predict(self, text):
        """
        Extract named entities from input text
        
        Args:
            text (str): Resume text content
            
        Returns:
            dict: Dictionary with extracted entities
        """
        # Tokenization
        tokenized_texts = []
        temp_token = []

        # Add [CLS] at the front
        temp_token.append('[CLS]')
        token_list = self.tokenizer.tokenize(text)
        
        for m, token in enumerate(token_list):
            temp_token.append(token)

        # Trim the token to fit the length requirement
        if len(temp_token) > self.max_len - 1:
            temp_token = temp_token[:self.max_len - 1]

        # Add [SEP] at the end
        temp_token.append('[SEP]')  

        tokenized_texts.append(temp_token)

        # Make id embedding  
        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=self.max_len, dtype='long', truncating='post', padding='post')
        
        # Make mask embedding
        attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        
        # Move tensors to device
        input_ids = input_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                token_type_ids=None,
                attention_mask=attention_masks,
            )
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        # Process predictions
        predict_results = logits.detach().cpu().numpy()
        results_arrays_soft = softmax(predict_results)  # Apply softmax
        result_list = np.argmax(results_arrays_soft, axis=-1)

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
        
        # Create a list to store token-tag pairs
        token_tag_pairs = []
        
        # Extract valid tokens and their tags
        for i in range(len(temp_token)):
            if i < len(attention_masks[0]) and attention_masks[0][i] > 0:
                token = temp_token[i]
                tag = self.idx2tag[result_list[0][i]]
                token_tag_pairs.append((token, tag))
        
        # Process token-tag pairs to reconstruct entities
        current_entity_type = None
        current_entity_tokens = []
        
        for i, (token, tag) in enumerate(token_tag_pairs):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]']:
                continue
                
            # Process tags
            if tag.startswith('B-'):  # Beginning of a multi-token entity
                # If we were building an entity, save it
                if current_entity_type and current_entity_tokens:
                    entity_text = self.reconstruct_entity_text(current_entity_tokens)
                    if current_entity_type in self.entity_dict:
                        result_json[self.entity_dict[current_entity_type]].append(entity_text)
                
                # Start new entity
                current_entity_type = tag[2:]
                current_entity_tokens = [token]
                
            elif tag.startswith('I-'):  # Inside of a multi-token entity
                if current_entity_type == tag[2:]:
                    current_entity_tokens.append(token)
                    
            elif tag.startswith('L-'):  # Last token of a multi-token entity
                if current_entity_type == tag[2:]:
                    current_entity_tokens.append(token)
                    # Save the completed entity
                    entity_text = self.reconstruct_entity_text(current_entity_tokens)
                    if current_entity_type in self.entity_dict:
                        result_json[self.entity_dict[current_entity_type]].append(entity_text)
                    # Reset for next entity
                    current_entity_type = None
                    current_entity_tokens = []
                    
            elif tag.startswith('U-'):  # Unit/single token entity
                entity_type = tag[2:]
                entity_text = token.replace('##', '')
                if entity_type in self.entity_dict:
                    result_json[self.entity_dict[entity_type]].append(entity_text)
                    
            elif tag == 'X':  # Subword token
                if current_entity_tokens:
                    current_entity_tokens.append(token)
        
        # In case the last entity wasn't closed with an L- tag
        if current_entity_type and current_entity_tokens:
            entity_text = self.reconstruct_entity_text(current_entity_tokens)
            if current_entity_type in self.entity_dict:
                result_json[self.entity_dict[current_entity_type]].append(entity_text)
        
        # Post-process the entities to clean them up
        for key in result_json:
            # Remove duplicates while preserving order
            unique_entities = []
            seen = set()
            for entity in result_json[key]:
                # Clean up entity text
                cleaned_entity = entity.strip()
                if cleaned_entity and cleaned_entity not in seen:
                    seen.add(cleaned_entity)
                    unique_entities.append(cleaned_entity)
            result_json[key] = unique_entities
        
        return result_json
    
    def predict_from_pdf(self, pdf_path, preprocessing=False):
        """
        Extract named entities from a PDF file
        
        Args:
            pdf_path (str): Path to PDF file
            preprocessing (bool): Whether to apply text preprocessing
            
        Returns:
            dict: Dictionary with extracted entities
        """
        text = self.pdf_to_text(pdf_path, preprocessing)
        return self.predict(text)


# Example usage:
if __name__ == "__main__":
    # Initialize the predictor with a model path
    predictor = ResumeNERPredictor(model_path="results/bert_ner_model.bin")
    
    # Example 1: Predict from text
    sample_text = """
    John Doe
    Software Engineer
    
    Education
    University of Technology, Bachelor of Computer Science, 2020
    
    Experience
    ABC Technologies, Senior Developer, 2020-Present
    XYZ Corp, Junior Developer, 2018-2020
    
    Skills
    Python, Java, Machine Learning, Deep Learning, NLP
    
    Contact
    email@example.com
    San Francisco, CA
    """
    
    results = predictor.predict(sample_text)
    print("Extracted information:")
    for entity_type, entities in results.items():
        if entities:
            print(f"{entity_type}: {', '.join(entities)}")
    
    # Example 2: Predict from PDF
    # results_pdf = predictor.predict_from_pdf("path/to/resume.pdf")
    # print("\nExtracted information from PDF:")
    # for entity_type, entities in results_pdf.items():
    #     if entities:
    #         print(f"{entity_type}: {', '.join(entities)}")