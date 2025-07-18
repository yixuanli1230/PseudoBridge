import json
import os
import time
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = OpenAI(api_key="your_api_key_here", timeout=30)  

SYSTEM_PROMPT = """
see the paper appendix
"""

def generate_code(docstring, pseudo_code, max_retries=3, backoff_factor=2):
    """
    Generate Python code using GPT-4o with error handling and retry mechanism
    
    Args:
    docstring (str): Function documentation string
    code (str): programming code
    max_retries (int): Maximum retry attempts
    backoff_factor (int): Exponential backoff multiplier
    
    Returns:
    str: Generated pseudo code
    """
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate pseudo code based on : the docstring \n{docstring} and the corresponding code:\n{code}"},
                ],
                temperature=0.3, 
                max_tokens=512,   
                stream=False
            )
            code = response.choices[0].message.content.strip()
            
            
            return code
        
        except Exception as e:
            attempt += 1
            wait_time = backoff_factor ** attempt
            logger.error(f"API call failed (Attempt {attempt}/{max_retries}): {str(e)}")
            logger.info(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    logger.error("Max retries exceeded. Returning empty code.")
    return "# Code generation failed"

def process_jsonl(input_file, output_file, start_line, end_line, save_interval, temp_output_dir):
    """
    Process JSONL file, generate code, and save results
    
    Args:
    input_file (str): Input file path
    output_file (str): Final output file path
    start_line (int): Starting line number
    end_line (int): Ending line number
    save_interval (int): Save frequency (records per file)
    temp_output_dir (str): Temporary output directory
    """
    # Ensure temp directory exists
    os.makedirs(temp_output_dir, exist_ok=True)
    
    processed_data = []   # Buffer for processed records
    temp_output_files = []  # Tracking temp files
    processed_count = 0    # Success counter
    file_count = 1         # Temp file numbering
    total_lines = 0        # For progress tracking

    try:
        # Calculate total lines for progress display
        with open(input_file, 'r', encoding='utf-8') as infile:
            total_lines = sum(1 for _ in infile)
        
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                # Skip lines before start position
                if line_num < start_line:
                    continue
                # Stop after end position
                if line_num > end_line:
                    break
                
                try:
                    data = json.loads(line.strip())
                    docstring = data.get("docstring", "")
                    code = data.get("code", "")
                    
                    # Validate required fields
                    if not docstring or not code:
                        logger.warning(f"Line {line_num} missing required fields")
                        continue
                    
                    # Generate code using GPT-4o
                    pseudo_code = generate_code(docstring, code)
                    
                    # Build output record
                    processed_data.append({
                        "docstring": docstring,
                        "pseudo_code": pseudo_code,
                        "code": code,
                    })
                    
                    processed_count += 1

                    if processed_count % 10 == 0 or line_num == end_line:
                        progress = min(100, int((line_num - start_line + 1) / (end_line - start_line + 1) * 100))
                        logger.info(f"Progress: {progress}% | Processed: {processed_count}/{end_line - start_line + 1} records")
                    
                    # Save temp files at specified intervals
                    if processed_count % save_interval == 0:
                        save_temp_file(processed_data, temp_output_dir, file_count, temp_output_files)
                        processed_data = []  # Clear buffer
                        file_count += 1
                
                except json.JSONDecodeError:
                    logger.error(f"Line {line_num} JSON decode error: {line}")
                except Exception as e:
                    logger.error(f"Line {line_num} processing failed: {str(e)}")
        
        # Save any remaining records in buffer
        if processed_data:
            save_temp_file(processed_data, temp_output_dir, file_count, temp_output_files)
        
        # Merge temp files into final output
        merge_temp_files(temp_output_files, output_file)
        logger.info(f"Successfully processed {processed_count} records. Output: {output_file}")
    
    except Exception as e:
        logger.critical(f"Processing aborted: {str(e)}")

def save_temp_file(data, temp_dir, file_count, file_list):
    """Save temporary JSONL file"""
    temp_file_path = os.path.join(temp_dir, f"temp_{file_count:04d}.jsonl")
    with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
        for record in data:
            temp_file.write(json.dumps(record, ensure_ascii=False) + '\n')
    file_list.append(temp_file_path)
    logger.info(f"Saved {len(data)} records to temp file: {temp_file_path}")

def merge_temp_files(temp_files, output_file):
    """Combine temporary files into final output"""
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for temp_file in temp_files:
            with open(temp_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
    logger.info(f"Merged {len(temp_files)} temp files into {output_file}")

if __name__ == "__main__":
    # Example paths - modify according to your environment
    input_file = '/path/to/input_data.jsonl'
    output_file = '/path/to/output_data.jsonl'
    temp_output_dir = '/path/to/temp_directory/'
    
    # Processing parameters
    start_line = 0        # First line to process (0-indexed)
    end_line = 100        # Last line to process
    save_interval = 10    # Records per temporary file
    
    # Execute processing pipeline
    logger.info("Starting data processing...")
    process_jsonl(input_file, output_file, start_line, end_line, save_interval, temp_output_dir)