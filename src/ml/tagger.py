import os
import re
import json
from typing import List, Dict, Any
import google.generativeai as genai
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class GeminiResumeNERTagger:
    """
    A class to perform Named Entity Recognition (NER) tagging on resume text using Gemini API.
    """
    
    def __init__(self, api_key):
        """
        Initialize the Gemini API client.
        
        Args:
            api_key: Gemini API key
        """
        # Configure the Gemini API with your API key
        genai.configure(api_key=api_key)
        
        # Define the model to use
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Define entity types to extract
        self.entity_types = [
            "Name",
            "Designation",
            "Companies worked at",
            "Location",
            "Email Address",
            "Skills",
            "College Name",
            "Graduation Year"
        ]
        
        # Download necessary NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

    def preprocess_text(self, text):
        """
        Preprocess the text to clean and normalize it.
        
        Args:
            text: The resume text string (already cleaned from HTML)
            
        Returns:
            Preprocessed text
        """
        text = text.replace("\n", " ")
        text = text.replace("\f", " ")
        return text

    def _create_prompt(self, text: str) -> str:
        """
        Create a prompt for the Gemini API.
        
        Args:
            text: The preprocessed resume text to analyze
            
        Returns:
            A formatted prompt string
        """
        prompt = f"""
        Extract the following entity types from this resume text:
        {', '.join(self.entity_types)}
        
        For each entity found, provide:
        1. The entity type
        2. The exact text of the entity
        3. The start position (character index) in the text
        4. The end position (character index) in the text
        
        Respond with a JSON object that has this structure:
        [
            {{
                "label": ["<entity_type>"],
                "points": [{{
                    "start": <start_position>,
                    "end": <end_position>,
                    "text": "<extracted_text>"
                }}]
            }},
            ...
        ]
        
        Here is the resume text to analyze:
        
        ```
        {text}
        ```
        
        Make sure to return ONLY the JSON without any other text or explanations.
        """
        return prompt

    def _fix_positions(self, text: str, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fix the positions of entities by finding the actual indices in the original text.
        
        Args:
            text: The original text
            annotations: The list of extracted entities with possibly incorrect positions
            
        Returns:
            A list of entities with corrected positions
        """
        fixed_annotations = []
        
        for annotation in annotations:
            for point in annotation["points"]:
                entity_text = point["text"]
                
                # Find the actual position in the text
                start_pos = text.find(entity_text)
                
                # If found, add with correct positions
                if start_pos != -1:
                    end_pos = start_pos + len(entity_text)
                    fixed_annotations.append({
                        "label": annotation["label"],
                        "points": [{
                            "start": start_pos,
                            "end": end_pos,
                            "text": entity_text
                        }]
                    })
                    
                # If exact match not found, try cleaning and matching again
                else:
                    # Clean entity text (remove extra spaces, newlines)
                    clean_entity = re.sub(r'\s+', ' ', entity_text).strip()
                    
                    # Try finding the cleaned entity
                    start_pos = text.find(clean_entity)
                    if start_pos != -1:
                        end_pos = start_pos + len(clean_entity)
                        fixed_annotations.append({
                            "label": annotation["label"],
                            "points": [{
                                "start": start_pos,
                                "end": end_pos,
                                "text": clean_entity
                            }]
                        })
                    # If still not found, try case-insensitive matching as a fallback
                    else:
                        text_lower = text.lower()
                        entity_lower = clean_entity.lower()
                        start_pos = text_lower.find(entity_lower)
                        if start_pos != -1:
                            # Find the actual text from the original using the position
                            actual_text = text[start_pos:start_pos + len(entity_lower)]
                            end_pos = start_pos + len(actual_text)
                            fixed_annotations.append({
                                "label": annotation["label"],
                                "points": [{
                                    "start": start_pos,
                                    "end": end_pos,
                                    "text": actual_text
                                }]
                            })
        
        return fixed_annotations

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts entities from the resume text using Gemini API
        
        Args:
            text: The cleaned text from a resume
            
        Returns:
            A list of dictionaries containing entity annotations
        """
        # Preprocess the text
        preprocessed_text = self.preprocess_text(text)
        
        # Create the prompt with preprocessed text
        prompt = self._create_prompt(preprocessed_text)
        
        # Call the Gemini API
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text
            
            # Extract the JSON part from the response if needed
            json_match = re.search(r'```json\n(.*?)\n```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            
            # Clean any remaining markdown code blocks
            result_text = re.sub(r'```.*?\n', '', result_text)
            result_text = re.sub(r'```', '', result_text)
            
            # Parse the JSON response
            annotations = json.loads(result_text)
            
            # Fix the positions to match the original text (not the preprocessed one)
            annotations = self._fix_positions(text, annotations)
            
            return annotations
        
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return []
    
    def format_output(self, text: str, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Formats the output similar to the provided example
        
        Args:
            text: The original text
            annotations: The list of extracted entities
            
        Returns:
            A formatted dictionary with content and annotations
        """
        return {
            "content": text,
            "annotation": annotations,
            "extras": None
        }
    
    def process_resume(self, text: str) -> Dict[str, Any]:
        """
        Main method to process a resume text
        
        Args:
            text: The cleaned text from a resume
            
        Returns:
            A formatted dictionary with content and annotations
        """
        annotations = self.extract_entities(text)
        return self.format_output(text, annotations)


def write_ndjson(data, output_file):
    """
    Write data to a Newline Delimited JSON (NDJSON) file.
    
    Args:
        data: List of dictionaries to write
        output_file: Path to the output file
    """
    with open(output_file, 'w') as f:
        for item in data:
            json_line = json.dumps(item)
            f.write(json_line + '\n')


def main():
    # Get API key from environment variable (more secure than hardcoding)
    api_key = 'AIzaSyBVwXNAK3z7ntUj7_auiRT4oYBvbwh4qV4'
    
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
    
    # Read data from CSV file
    raw_data = pd.read_csv("./dataset/kaggle/Resume/Resume.csv")
    raw_data = raw_data['Resume_str'].values.tolist()
    raw_data = raw_data[:200]  # Just process first 200 for testing

    # Process the resumes
    tagger = GeminiResumeNERTagger(api_key)
    results = [tagger.process_resume(x) for x in raw_data]
    
    # Write the results to an NDJSON file
    output_file = "resume_tags_output.json"
    write_ndjson(results, output_file)
    
    print(f"Results successfully written to {output_file}")


if __name__ == "__main__":
    main()