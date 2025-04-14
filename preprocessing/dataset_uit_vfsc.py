import pandas as pd
import os
import glob
import csv

def process_txt_to_csv():
    base_input_dir = os.path.join(os.path.dirname(__file__), 'DATASET_UIT_VFSC')
    base_output_dir = os.path.join(os.path.dirname(__file__), 'DATASET_UIT_VFSC_PROCESSED')
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    subdirs = ['train', 'val', 'test']
    
    for subdir in subdirs:
        data_dir = os.path.join(base_input_dir, subdir)
        output_file = os.path.join(base_output_dir, f'{subdir}.csv')
        
        if not os.path.exists(data_dir):
            continue
            
        sents_path = os.path.join(data_dir, 'sents.txt')
        sentiments_path = os.path.join(data_dir, 'sentiments.txt')
        
        if not os.path.exists(sents_path) or not os.path.exists(sentiments_path):
            continue
            
        with open(sents_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f.readlines() if line.strip()]
            
        with open(sentiments_path, 'r', encoding='utf-8') as f:
            sentiments = [line.strip() for line in f.readlines() if line.strip()]
            
        if len(sentences) != len(sentiments):
            continue
            
        data = []
        for sentence, sentiment in zip(sentences, sentiments):
            data.append({
                'sentence': sentence,
                'sentiment': sentiment
            })
        
        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            print(f"Saved {len(df)} records to {output_file}")
        else:
            print(f"No data found in {data_dir}")

def process_existing_csvs():
    output_dir = os.path.join(os.path.dirname(__file__), 'DATASET_UIT_VFSC_PROCESSED')
    os.makedirs(output_dir, exist_ok=True)
    
    for csv_file in glob.glob(os.path.join(os.path.dirname(__file__), 'DATASET_UIT_VFSC', '*.csv')):
        filename = os.path.basename(csv_file)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Processing {csv_file}...")
        df = pd.read_csv(csv_file)
        
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['sentence', 'sentiment']
            
            df.to_csv(output_path, index=False)
            print(f"Saved processed file to {output_path}")

def clean_processed_files():
    processed_dir = os.path.join(os.path.dirname(__file__), 'DATASET_UIT_VFSC_PROCESSED')
    
    for csv_file in glob.glob(os.path.join(processed_dir, '*.csv')):
        print(f"Cleaning {csv_file}...")
        
        try:
            df = pd.read_csv(csv_file, encoding='utf-8', quoting=csv.QUOTE_ALL)
            
            if 'sentence' in df.columns:
                df['sentence'] = df['sentence'].astype(str).apply(lambda x: x.replace('"', ''))
                
                temp_file = csv_file + ".temp"
                df.to_csv(temp_file, index=False, encoding='utf-8', quoting=csv.QUOTE_NONE, escapechar='\\')
                
                os.remove(csv_file)
                os.rename(temp_file, csv_file)
                print(f"Cleaned and saved {csv_file}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

def main():
    print("BEFORE PROCESSING:\n")

    process_txt_to_csv()
    process_existing_csvs()
    clean_processed_files()

    print("\nAFTER PROCESSING:")

if __name__ == "__main__":
    main() 