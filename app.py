from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import tempfile
import urllib.request

app = Flask(__name__)

id2label = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}

model = None
tokenizer = None
model_loaded = False

def init_model():
    global model, tokenizer, model_loaded
    try:
        print("Bắt đầu tải mô hình...")
        # Tải tokenizer
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        print("Đã tải tokenizer")
        
        # Tải mô hình cơ bản
        model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
        print("Đã tải mô hình cơ bản")
        
        # Tạo thư mục tạm phù hợp với Railway
        temp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Tải checkpoint từ Hugging Face
            checkpoint_url = "https://huggingface.co/NgoQuangNghia111003/phobert/resolve/main/phobert.ckpt"
            print(f"Đang tải checkpoint từ {checkpoint_url}")
            
            temp_file = os.path.join(temp_dir, "checkpoint_temp.ckpt")
            urllib.request.urlretrieve(checkpoint_url, temp_file)
            
            file_size = os.path.getsize(temp_file) / (1024 * 1024)
            print(f"Đã tải checkpoint: {file_size:.2f} MB")
            
            # Tải trọng số
            checkpoint = torch.load(temp_file, map_location='cpu', weights_only=False)
            
            if 'state_dict' in checkpoint:
                state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() 
                             if k.startswith('model.')}
                model.load_state_dict(state_dict)
                print("Đã tải trọng số thành công!")
                model_loaded = True
            else:
                print("Cấu trúc checkpoint không như mong đợi")
                
            # Xóa file tạm sau khi dùng
            os.remove(temp_file)
            
        except Exception as e:
            print(f"Lỗi khi tải checkpoint: {e}")
            print("Sử dụng mô hình cơ bản thay thế")
        
        # Đặt chế độ đánh giá
        model.eval()
        return True
        
    except Exception as e:
        print(f"Lỗi tổng thể: {e}")
        return False

def predict(text):
    if not text:
        return None
        
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
            "data": result,
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "data": None
        }), 500

# Khởi tạo mô hình khi khởi động
print("Khởi tạo ứng dụng...")
init_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
