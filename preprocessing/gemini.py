import os
import google.generativeai as genai
import time
import csv
import json
import glob
import re
from config import GOOGLE_API_KEY, GEMINI_MODEL
import concurrent.futures
import random
import pandas as pd

BATCH_SIZE = 50
MAX_WORKERS = 2
REQUEST_DELAY = 3
QUOTA_RETRY_DELAY = 60
MAX_RETRIES = 5
CHECKPOINT_FILE = 'generation_checkpoint.json'

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'GEMINI')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'train.csv')

genai.configure(api_key=GOOGLE_API_KEY)

def extract_retry_delay(error_message):
    match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', str(error_message))
    if match:
        return int(match.group(1))
    return QUOTA_RETRY_DELAY

def generate_data_batch(category, sentiment_label, batch_num, retries=0):
    if retries > MAX_RETRIES:
        print(f"Quá số lần thử lại cho batch {batch_num} của {category}")
        return []
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        if sentiment_label == 0:
            prompt = f"""
            Bạn là một người tạo ra các câu phản hồi CỰC KỲ tiêu cực về giáo dục bằng tiếng Việt. 
            Hãy tạo ra {BATCH_SIZE} câu phản hồi TIÊU CỰC MẠNH MẼ của học sinh, sinh viên về giáo viên/khóa học với chủ đề cảm xúc sau:
            {category}
            LƯU Ý:
            1. Các câu đều phải là quan điểm của học sinh, sinh viên ĐẦY BẤT MÃN VÀ KHÔNG HÀI LÒNG
            2. KHÔNG TẠO những lời góp ý mang tính xây dựng, phải là những lời phàn nàn, chỉ trích gay gắt
            3. Mỗi câu phải có độ dài từ 50 đến 100 ký tự
            4. Tất cả câu đều có nhãn cảm xúc là 0 (cảm xúc tiêu cực)
            5. Không được sử dụng dấu nháy kép (") trong các câu
            6. Mỗi câu phải khác nhau, không được trùng lặp
            7. Các câu phải rất tiêu cực và bộc lộ cảm xúc mạnh mẽ (bực tức, giận dữ, thất vọng sâu sắc)
            Hãy chỉ trả về {BATCH_SIZE} dòng với định dạng chính xác này (không có bất kỳ văn bản bổ sung nào):
            <câu>,0
            """
        else:
            prompt = f"""
            Bạn là một người tạo ra các câu phản hồi CỰC KỲ tích cực về giáo dục bằng tiếng Việt. 
            Hãy tạo ra {BATCH_SIZE} câu phản hồi TÍCH CỰC MẠNH MẼ của học sinh, sinh viên về giáo viên/khóa học với chủ đề cảm xúc sau:
            {category}
            LƯU Ý:
            1. Các câu đều phải là quan điểm của học sinh, sinh viên ĐẦY PHẤN KHỞI VÀ HÀI LÒNG
            2. Mỗi câu phải thể hiện sự ngưỡng mộ, cảm phục, biết ơn hoặc yêu thích
            3. Mỗi câu phải có độ dài từ 50 đến 100 ký tự
            4. Tất cả câu đều có nhãn cảm xúc là 2 (cảm xúc tích cực)
            5. Không được sử dụng dấu nháy kép (") trong các câu
            6. Mỗi câu phải khác nhau, không được trùng lặp
            7. Các câu phải rất tích cực và bộc lộ cảm xúc mạnh mẽ (biết ơn, vui mừng, phấn khởi)
            Hãy chỉ trả về {BATCH_SIZE} dòng với định dạng chính xác này (không có bất kỳ văn bản bổ sung nào):
            <câu>,2
            """
        response = model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        result = []
        for line in lines:
            if ',' in line:
                try:
                    sentence, sentiment = line.rsplit(',', 1)
                    sentence = sentence.replace('"', '')
                    sentiment = sentiment.strip()
                    if sentiment == str(sentiment_label) and 50 <= len(sentence) <= 100:
                        result.append([sentence, int(sentiment)])
                except Exception:
                    continue
        return result
    except Exception as e:
        error_message = str(e)
        print(f"Error in batch {batch_num} for {category}: {error_message}")
        if "429" in error_message or "quota" in error_message.lower():
            wait_time = extract_retry_delay(error_message)
            print(f"Quota exceeded! Waiting for {wait_time} seconds before retry...")
            time.sleep(wait_time)
            return generate_data_batch(category, sentiment_label, batch_num, retries + 1)
        time.sleep(REQUEST_DELAY * 2)
        return generate_data_batch(category, sentiment_label, batch_num, retries + 1)

def save_checkpoint(progress):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def process_category(category, sentiment_label, count=2000, skip_existing=True):
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sentence', 'sentiment'])
    checkpoint = load_checkpoint()
    category_key = f"{category}_{sentiment_label}"
    start_count = checkpoint.get(category_key, 0) if skip_existing else 0
    if start_count >= count:
        print(f"Skipping {category} - Already completed {start_count}/{count}")
        return {"category": category, "sentiment": sentiment_label, "count": start_count}
    print(f"Processing {category} from {start_count}/{count}")
    csv_file = open(OUTPUT_FILE, 'a', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE, escapechar='\\')
    generated_count = start_count
    batch_num = start_count // BATCH_SIZE
    try:
        while generated_count < count:
            print(f"{category}: Processing batch {batch_num+1} - Current: {generated_count}/{count}")
            batch_result = generate_data_batch(category, sentiment_label, batch_num)
            new_count = 0
            for item in batch_result:
                if generated_count < count:
                    csv_writer.writerow(item)
                    generated_count += 1
                    new_count += 1
            csv_file.flush()
            checkpoint[category_key] = generated_count
            save_checkpoint(checkpoint)
            print(f"{category}: Batch {batch_num+1} added {new_count} items - Total: {generated_count}/{count}")
            batch_num += 1
            delay = REQUEST_DELAY + random.uniform(0.5, 2.0)
            time.sleep(delay)
    finally:
        csv_file.close()
    return {"category": category, "sentiment": sentiment_label, "count": generated_count}

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

def create_test_val_csv_from_train():
    processed_dir = os.path.join(os.path.dirname(__file__), 'GEMINI')
    train_csv_path = os.path.join(processed_dir, 'train.csv')
    val_csv_path = os.path.join(processed_dir, 'val.csv')
    test_csv_path = os.path.join(processed_dir, 'test.csv')

    try:
        full_df = pd.read_csv(train_csv_path, encoding='utf-8', quoting=csv.QUOTE_ALL, escapechar='\\', on_bad_lines='skip')
        print(f"Đọc thành công {train_csv_path}")
    except Exception as e:
        print(f"Failed to read {train_csv_path}: {e}")
        return

    train_size = int(0.8 * len(full_df))
    val_size = int(0.1 * len(full_df))

    train_df = full_df[:train_size]
    train_df.to_csv(train_csv_path, index=False, encoding='utf-8')

    val_df = full_df[train_size:train_size + val_size]
    val_df.to_csv(val_csv_path, index=False, encoding='utf-8')
    print(f"Đã sinh dữ liệu val tại: {val_csv_path}")

    test_df = full_df[train_size + val_size:]
    test_df.to_csv(test_csv_path, index=False, encoding='utf-8')
    print(f"Đã sinh dữ liệu test tại: {test_csv_path}")

    paraphrased_val = []
    paraphrased_test = []

    for _, row in val_df.iterrows():
        sentiment_text = {0: "tiêu cực", 1: "trung lập", 2: "tích cực"}.get(row['sentiment'], "trung lập")
        new_sentence = generate_paraphrased_sentence(row['sentence'], sentiment_text)
        paraphrased_val.append({'sentence': new_sentence, 'sentiment': row['sentiment']})

    for _, row in test_df.iterrows():
        sentiment_text = {0: "tiêu cực", 1: "trung lập", 2: "tích cực"}.get(row['sentiment'], "trung lập")
        new_sentence = generate_paraphrased_sentence(row['sentence'], sentiment_text)
        paraphrased_test.append({'sentence': new_sentence, 'sentiment': row['sentiment']})

    val_paraphrased_df = pd.DataFrame(paraphrased_val)
    val_paraphrased_df.to_csv(val_csv_path, index=False, encoding='utf-8')
    print(f"Đã sinh dữ liệu paraphrased cho val tại: {val_csv_path}")

    test_paraphrased_df = pd.DataFrame(paraphrased_test)
    test_paraphrased_df.to_csv(test_csv_path, index=False, encoding='utf-8')
    print(f"Đã sinh dữ liệu paraphrased cho test tại: {test_csv_path}")

def main():
    categories = {
        "Bất mãn sâu sắc (về chương trình học, nội quy trường, cơ sở vật chất)": 0,
        "Tức giận dữ dội (biểu đạt sự khó chịu tột độ với giảng viên, bạn học)": 0,
        "Buồn bã cùng cực (nỗi thất vọng, mất mát trong học tập)": 0,
        "Sợ hãi/lo lắng cực độ (về tương lai, kết quả học tập)": 0,
        "Chỉ trích nặng nề (đối với cách dạy, phương pháp đánh giá)": 0,
        "Thất vọng sâu sắc (về kỳ vọng không được đáp ứng)": 0,
        "Cô đơn tột cùng (cảm giác bị cô lập, thiếu kết nối)": 0,
        "Oán giận cay nghiệt (về cơ sở vật chất và môi trường học tập tồi tàn)": 0,
        "Phẫn nộ không kiềm chế (về chính sách học phí và tài chính của nhà trường)": 0,
        "Biết ơn sâu sắc (với giáo viên về sự tận tâm và kiến thức truyền đạt)": 2,
        "Hạnh phúc tràn đầy (với trải nghiệm học tập bổ ích và thú vị)": 2,
        "Ngưỡng mộ chân thành (với chuyên môn và phương pháp giảng dạy của giáo viên)": 2,
        "Hứng khởi mạnh mẽ (với nội dung bài học và cách truyền đạt sáng tạo)": 2,
        "Phấn chấn tích cực (với môi trường học tập và bạn học đầy hỗ trợ)": 2,
        "Tự tin vững vàng (với kiến thức và kỹ năng đã học được sau khóa học)": 2,
        "Tự hào to lớn (về thành tích học tập và sự tiến bộ của bản thân)": 2,
        "Yêu thích sâu đậm (đối với môn học và chuyên ngành đang theo học)": 2,
        "Trân trọng sâu sắc (đối với cơ sở vật chất hiện đại và đầy đủ của nhà trường)": 2,
        "Hài lòng tuyệt đối (với chính sách học phí và hỗ trợ tài chính của nhà trường)": 2
    }
    results = []
    checkpoint = load_checkpoint()
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sentence', 'sentiment'])
    for category, sentiment in categories.items():
        category_key = f"{category}_{sentiment}"
        completed = checkpoint.get(category_key, 0)
        target = 2000
        if completed >= target:
            print(f"Skipping {category} - Already completed")
            results.append({"category": category, "sentiment": sentiment, "count": completed})
            continue
        result = process_category(category, sentiment, count=target)
        results.append(result)
    total_generated = sum(result["count"] for result in results)
    print(f"All processing complete. Total generated: {total_generated} sentences")

    create_test_val_csv_from_train()


if __name__ == "__main__":
    main() 