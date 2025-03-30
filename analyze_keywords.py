import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from main import predict_sentiment
import spacy
import argparse

# Tải mô hình ngôn ngữ tiếng Việt. Cần cài đặt thêm bằng: python -m spacy download xx_ent_wiki_sm
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except:
    print("Cần cài đặt mô hình spaCy cho tiếng Việt: python -m spacy download xx_ent_wiki_sm")
    exit(1)

def extract_keywords(text):
    """Trích xuất từ khóa từ văn bản"""
    doc = nlp(text)
    
    # Lấy các từ có ý nghĩa (loại bỏ stopwords, dấu câu)
    keywords = [token.text.lower() for token in doc 
               if not token.is_stop and not token.is_punct and len(token.text) > 1]
    
    # Lấy các thực thể có tên
    entities = [ent.text for ent in doc.ents]
    
    return keywords, entities

def analyze_text_file(file_path):
    """Phân tích file văn bản và tạo báo cáo chi tiết"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    results = []
    all_keywords = []
    all_entities = []
    sentiments = {"Tích cực": 0, "Trung tính": 0, "Tiêu cực": 0}
    
    for line in lines:
        # Phân tích cảm xúc
        sentiment_result = predict_sentiment(line)
        
        # Trích xuất từ khóa
        keywords, entities = extract_keywords(line)
        
        all_keywords.extend(keywords)
        all_entities.extend(entities)
        sentiments[sentiment_result["sentiment"]] += 1
        
        results.append({
            "text": line,
            "sentiment": sentiment_result["sentiment"],
            "confidence": sentiment_result["confidence"],
            "keywords": keywords,
            "entities": entities
        })
    
    # Phân tích từ khóa
    keyword_counts = Counter(all_keywords)
    entity_counts = Counter(all_entities)
    
    # Tạo DataFrame từ kết quả
    df = pd.DataFrame(results)
    
    # Tính toán thống kê
    sentiment_stats = df["sentiment"].value_counts().to_dict()
    top_keywords = keyword_counts.most_common(20)
    top_entities = entity_counts.most_common(10)
    
    # Phân tích từ khóa theo cảm xúc
    keywords_by_sentiment = {
        "Tích cực": [],
        "Trung tính": [],
        "Tiêu cực": []
    }
    
    for _, row in df.iterrows():
        for keyword in row["keywords"]:
            keywords_by_sentiment[row["sentiment"]].append(keyword)
    
    keyword_sentiment = {
        "Tích cực": Counter(keywords_by_sentiment["Tích cực"]).most_common(10),
        "Trung tính": Counter(keywords_by_sentiment["Trung tính"]).most_common(10),
        "Tiêu cực": Counter(keywords_by_sentiment["Tiêu cực"]).most_common(10)
    }
    
    # Tạo báo cáo
    report = {
        "total_texts": len(results),
        "sentiment_stats": sentiment_stats,
        "top_keywords": top_keywords,
        "top_entities": top_entities,
        "keyword_by_sentiment": keyword_sentiment,
        "detailed_results": df
    }
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Phân tích cảm xúc và từ khóa văn bản tiếng Việt")
    parser.add_argument("file", help="Đường dẫn tới file văn bản cần phân tích")
    parser.add_argument("--output", "-o", help="Đường dẫn lưu file báo cáo CSV")
    
    args = parser.parse_args()
    
    print(f"Đang phân tích file: {args.file}")
    report = analyze_text_file(args.file)
    
    print(f"\nKết quả phân tích cho {report['total_texts']} văn bản:")
    print(f"- Tích cực: {report['sentiment_stats'].get('Tích cực', 0)}")
    print(f"- Trung tính: {report['sentiment_stats'].get('Trung tính', 0)}")
    print(f"- Tiêu cực: {report['sentiment_stats'].get('Tiêu cực', 0)}")
    
    print("\nTop 10 từ khóa xuất hiện nhiều nhất:")
    for keyword, count in report["top_keywords"][:10]:
        print(f"- {keyword}: {count}")
    
    if args.output:
        report["detailed_results"].to_csv(args.output, index=False)
        print(f"\nĐã lưu báo cáo chi tiết vào: {args.output}")

if __name__ == "__main__":
    main() 