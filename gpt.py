import pandas as pd
import os
import glob
import csv
import google.generativeai as genai
from config import GOOGLE_API_KEY, GEMINI_MODEL
genai.configure(api_key=GOOGLE_API_KEY)

def process_existing_csvs():
    output_dir = os.path.join(os.path.dirname(__file__), 'GPT_PROCESSED')
    os.makedirs(output_dir, exist_ok=True)
    
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    for csv_file in glob.glob(os.path.join(os.path.dirname(__file__), 'GPT', '*.csv')):
        filename = os.path.basename(csv_file)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Processing {csv_file}...")
        df = pd.read_csv(csv_file)
        
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['sentence', 'sentiment']
            df['sentiment'] = df['sentiment'].map(sentiment_map)
            df.to_csv(output_path, index=False)
            print(f"Saved processed file to {output_path}")

def clean_processed_files():
    processed_dir = os.path.join(os.path.dirname(__file__), 'GPT_PROCESSED')
    
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

def generate_paraphrased_sentence(original_sentence, sentiment_label):
    prompt = f"""Dưới đây là một câu biểu thị cảm xúc {sentiment_label}. Hãy viết lại câu này theo cách khác nhưng giữ nguyên nghĩa và cảm xúc đó.
Câu gốc: "{original_sentence}"
Câu viết lại:"""
    
    try:
        response = genai.generate_text(
            model=GEMINI_MODEL,
            prompt=prompt,
            temperature=0.7,
            max_output_tokens=100
        )
        return response.text.strip().replace('"', '')
    except Exception as e:
        print(f"Error generating sentence: {e}")
        return original_sentence

def create_test_csv_from_train():
    processed_dir = os.path.join(os.path.dirname(__file__), 'GPT_PROCESSED')
    train_csv_path = os.path.join(processed_dir, 'train.csv')
    test_csv_path = os.path.join(processed_dir, 'test.csv')

    try:
        full_df = pd.read_csv(train_csv_path, encoding='utf-8', quoting=csv.QUOTE_ALL, escapechar='\\', on_bad_lines='skip')
        print(f"Đọc thành công {train_csv_path}")
    except Exception as e:
        print(f"Failed to read {train_csv_path}: {e}")
        return

    test_size = int(0.1 * len(full_df))
    test_df = full_df[-test_size:]
    test_df.to_csv(test_csv_path, index=False, encoding='utf-8')
    print(f"Đã sinh dữ liệu test tại: {test_csv_path}")

    paraphrased_test = []

    for _, row in test_df.iterrows():
        sentiment_text = {0: "tiêu cực", 1: "trung lập", 2: "tích cực"}.get(row['sentiment'], "trung lập")
        new_sentence = generate_paraphrased_sentence(row['sentence'], sentiment_text)
        paraphrased_test.append({'sentence': new_sentence, 'sentiment': row['sentiment']})

    test_paraphrased_df = pd.DataFrame(paraphrased_test)
    test_paraphrased_df.to_csv(test_csv_path, index=False, encoding='utf-8')
    print(f"Đã sinh dữ liệu paraphrased cho test tại: {test_csv_path}")

def main():
    print("BEFORE PROCESSING:\n")
    process_existing_csvs()
    clean_processed_files()
    create_test_csv_from_train()
    print("\nAFTER PROCESSING:")

if __name__ == "__main__":
    main() 