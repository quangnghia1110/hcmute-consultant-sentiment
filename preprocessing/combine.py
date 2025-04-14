import os
import pandas as pd
import random
import string
import csv

def generate_id(index):
    """Generate ID with 5 digits starting from 10000 + 3 random letters"""
    num_part = str(10000 + index)
    letters_part = ''.join(random.choices(string.ascii_uppercase, k=3))
    return f"{num_part}{letters_part}"

# Create output directory
output_dir = "hcmute-consultant-sentiment/COMBINED_PROCESSED"
os.makedirs(output_dir, exist_ok=True)

# Process each file explicitly
for filename in ["train.csv", "val.csv", "test.csv"]:
    input_path = f"hcmute-consultant-sentiment/COMBINED/{filename}"
    output_path = os.path.join(output_dir, filename)
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Warning: {filename} not found at {input_path}")
        continue
        
    print(f"Processing {filename}...")
    
    # Read the file with CSV reader to handle commas correctly
    rows = []
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for i, row in enumerate(reader):
            if len(row) >= 2:
                text = row[0]
                try:
                    label = int(row[1])
                except:
                    label = 1  # Default to neutral
                
                rows.append({
                    "id": generate_id(i),
                    "text": text,
                    "class": label
                })
    
    # Create dataframe
    df = pd.DataFrame(rows)
    
    # Write the CSV file manually to control the header format exactly
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        # Write header without quotes
        f.write("id,text,class\n")
        
        # Write data rows with proper quoting
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for _, row in df.iterrows():
            writer.writerow([row['id'], row['text'], row['class']])
    
    print(f"Saved {len(df)} rows to {output_path}")
    
    # Verify the first few lines
    with open(output_path, 'r', encoding='utf-8') as f:
        first_lines = [next(f) for _ in range(3)]
    print(f"File begins with: {first_lines[0].strip()}")
    if len(first_lines) > 1:
        print(f"First data row: {first_lines[1].strip()}")

print("All files processed!")