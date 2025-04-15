from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import time

app = Flask(__name__)

id2label = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "model/phobert-sentiment-epochepoch=08-val_f1val_f1=0.9445.ckpt")
model = None
tokenizer = None

def init_model():
    global model, tokenizer
    print("Bắt đầu tải mô hình...")
    start_time = time.time()
    try:
        # Tải tokenizer trước
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        print(f"Đã tải tokenizer sau {time.time() - start_time:.2f} giây")
        
        # Tải mô hình base
        model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
        print(f"Đã tải mô hình base sau {time.time() - start_time:.2f} giây")
        
        # Tải trọng số từ checkpoint
        print(f"Đang tải trọng số từ {CHECKPOINT_PATH}...")
        state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
        if 'state_dict' in state_dict:
            state_dict = {k.replace('model.', ''): v for k, v in state_dict['state_dict'].items() 
                        if k.startswith('model.')}
            model.load_state_dict(state_dict)
        
        # Chuyển sang chế độ đánh giá
        model.eval()
        print(f"Đã tải xong mô hình sau {time.time() - start_time:.2f} giây")
        return True
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return False

def get_model():
    global model, tokenizer
    if model is None:
        success = init_model()
        if not success:
            raise Exception("Không thể tải mô hình")
    return model, tokenizer

def predict(text):
    if not text:
        return None
    
    # Lấy mô hình khi cần (lazy loading)
    model, tokenizer = get_model()
        
    text = ' '.join(text.strip().split())
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0].tolist()
        
    predicted_class = probs.index(max(probs))
    second_max = sorted(probs, reverse=True)[1]
    
    if max(probs) - second_max < 0.1 and probs.index(second_max) == 1:
        predicted_class = 1
    
    return {"classify": id2label[predicted_class]}

@app.route('/sentiment', methods=['GET'])
def analyze_sentiment():
    text = request.args.get('text', '').strip()
    if not text:
        return jsonify({
            "status": "error",
            "message": "Thiếu dữ liệu đầu vào",
            "data": None
        }), 400
    
    try:
        result = predict(text)
        return jsonify({
            "status": "success", 
            "message": "Phân tích cảm xúc thành công",
            "data": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "data": None
        }), 500

# Khởi tạo mô hình khi khởi động Flask với Gunicorn
# Việc này tăng thời gian khởi động nhưng đảm bảo mô hình được tải trước khi có request
try:
    init_model()
except Exception as e:
    print(f"Lỗi khi khởi tạo mô hình: {e}")
    print("Ứng dụng sẽ tải mô hình khi có request đầu tiên")

if __name__ == "__main__":
    # Đảm bảo mô hình đã được tải khi chạy trực tiếp
    if not init_model():
        print("Cảnh báo: Mô hình chưa được tải, sẽ tải khi có request")
    app.run(host="0.0.0.0", port=5000, debug=False) 