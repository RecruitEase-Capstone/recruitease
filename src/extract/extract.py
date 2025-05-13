
import os
import io
import zipfile
import PyPDF2
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def extract_zip():
    """
    Extract ZIP file contents directly
    
    Returns:
        list: List PDF files
    """
    # Hide the root Tkinter window
    Tk().withdraw()

    # Prompt user to select a ZIP file
    zip_file_path = askopenfilename(title="Select a ZIP file", filetypes=[("ZIP files", "*.zip")])
    if not zip_file_path:
        print("No file selected.")
        return []

    # List to store PDF files
    pdf_files = []

    try:
        # Open the ZIP file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Find and extract only PDF files
            pdf_names = [name for name in zip_ref.namelist() if name.lower().endswith('.pdf')]
            
            # Extract PDFs
            for pdf_name in pdf_names:
                with zip_ref.open(pdf_name) as pdf_file:
                    # Read PDF file
                    pdf_bytes = pdf_file.read()
                    pdf_files.append({
                        'filename': os.path.basename(pdf_name),
                        'file_bytes': pdf_bytes
                    })
        
        print(f"{len(pdf_files)} PDF files extracted")
        return pdf_files
    
    except zipfile.BadZipFile:
        print("The selected file is not a valid ZIP file.")
        return []

def extract_pdf_to_texts(pdf_files):
    """
    Extract text from PDF files
    
    Args:
        pdf_files (list): List of PDF files
    
    Returns:
        list: List of dictionaries with filename and extracted text
    """
    # List to store extracted text
    extracted_data = []

    # Counter for successful and failed extractions
    successful_extractions = 0
    failed_extractions = 0

    # Process each PDF file
    for pdf_info in pdf_files:
        try:
            # Create PDF reader from bytes
            pdf_file = io.BytesIO(pdf_info['file_bytes'])
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"
            
            # Add to extracted data
            extracted_data.append({
                'filename': pdf_info['filename'],
                'text': full_text.strip()
            })
            successful_extractions += 1
        
        except Exception as e:
            failed_extractions += 1
    
    # Save to JSON
    try:
        # Ensure extracted_data directory exists
        os.makedirs('extracted_data', exist_ok=True)
        
        json_path = os.path.join('extracted_data', 'extracted_pdf_texts.json')
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            # Use indent=4 for more readable JSON
            json.dump(extracted_data, jsonfile, ensure_ascii=False, indent=4)
        
        print(f"\nJSON file saved to {json_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")
    
    # Summary of extraction
    print(f"\nExtraction Summary:")
    print(f"Total PDF files: {len(pdf_files)}")
    print(f"Successfully extracted: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    
    return extracted_data

def main():
    # Extract ZIP files
    pdf_files = extract_zip()
    
    # Extract texts from PDFs if any were found
    if pdf_files:
        extracted_data = extract_pdf_to_texts(pdf_files)
    else:
        print("No PDF files to extract text from.")

if __name__ == "__main__":
    main()
