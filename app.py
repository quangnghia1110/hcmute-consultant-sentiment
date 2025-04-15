from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
import os
import gc
import urllib.request

# Tắt cảnh báo, tiết kiệm bộ nhớ
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)

id2label = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
CHECKPOINT_URL = "https://huggingface.co/NgoQuangNghia111003/phobert/resolve/main/phobert.ckpt"

model = None
tokenizer = None

def get_model():
    global model, tokenizer
    
    if model is None:
        try:
            print("Đang tải mô hình của bạn...")
            
            # Tải tokenizer
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            
            # Tải model base với tối ưu
            model = AutoModelForSequenceClassification.from_pretrained(
                "vinai/phobert-base", 
                num_labels=3,
                torchscript=True,
                low_cpu_mem_usage=True
            )
            
            # Tạo thư mục tạm
            temp_dir = "model_cache"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, "checkpoint.ckpt")
            
            # Tải checkpoint
            urllib.request.urlretrieve(CHECKPOINT_URL, temp_file)
            
            # Tải trọng số (memory mapping để không tải toàn bộ vào RAM)
            checkpoint = torch.load(temp_file, map_location='cpu', weights_only=True)
            
            if 'state_dict' in checkpoint:
                # Tải từng tensor một để tiết kiệm bộ nhớ
                state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if key.startswith('model.'):
                        new_key = key.replace('model.', '')
                        state_dict[new_key] = value
                
                model.load_state_dict(state_dict)
                print("Đã tải trọng số thành công!")
                
                # Xóa state_dict để giải phóng bộ nhớ
                del checkpoint, state_dict
            
            # Xóa file tạm
            os.remove(temp_file)
            
            # Đặt mode evaluation và tối ưu
            model.eval()
            
            # Thu gom rác
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            # Sử dụng model dự phòng nếu lỗi
            if model is None:
                print("Sử dụng model dự phòng...")
                model = AutoModelForSequenceClassification.from_pretrained(
                    "vinai/phobert-base", 
                    num_labels=3
                )
                model.eval()
    
    return model, tokenizer

@app.route('/sentiment', methods=['GET'])
def analyze_sentiment():
    text = request.args.get('text', '').strip()
    if not text:
        return jsonify({"status": "error", "message": "Thiếu dữ liệu"}), 400
    
    try:
        model, tokenizer = get_model()
        
        # Dự đoán với giới hạn bộ nhớ
        text = ' '.join(text.strip().split())
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0]
            probs = F.softmax(logits, dim=-1)[0].tolist()
        
        predicted_class = probs.index(max(probs))
        
        # Giải phóng bộ nhớ
        del inputs, outputs, logits
        gc.collect()
        
        return jsonify({
            "status": "success", 
            "data": {"classify": id2label[predicted_class]}
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# KHÔNG tải mô hình khi khởi động
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)