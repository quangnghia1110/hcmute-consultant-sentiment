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
model_loaded = False  # Biến theo dõi trạng thái tải mô hình

def init_model():
    global model, tokenizer, model_loaded
    print("Bắt đầu tải mô hình...")
    start_time = time.time()
    try:
        # Tải tokenizer
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        print(f"Đã tải tokenizer sau {time.time() - start_time:.2f} giây")
        
        # Tải mô hình base
        model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
        print(f"Đã tải mô hình base sau {time.time() - start_time:.2f} giây")
        
        # Kiểm tra nếu file checkpoint tồn tại
        if os.path.exists(CHECKPOINT_PATH):
            file_size = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)  # MB
            print(f"File checkpoint tồn tại, kích thước: {file_size:.2f} MB")
            
            # Nếu file quá nhỏ, có thể chỉ là file pointer của Git LFS
            if file_size < 1:  # Nếu nhỏ hơn 1MB, có thể là file pointer
                print("CẢNH BÁO: File checkpoint có kích thước quá nhỏ, có thể là file pointer của Git LFS!")
        else:
            print(f"CẢNH BÁO: Không tìm thấy file checkpoint tại {CHECKPOINT_PATH}")
        
        # Thử tải checkpoint
        try:
            print(f"Đang thử tải checkpoint...")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            
            # Kiểm tra cấu trúc checkpoint
            if 'state_dict' in checkpoint:
                state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() 
                            if k.startswith('model.')}
                model.load_state_dict(state_dict)
                print("Đã tải checkpoint thành công!")
                model_loaded = True
            else:
                print(f"Cấu trúc checkpoint không như mong đợi, các khóa có: {list(checkpoint.keys())}")
                # Sử dụng mô hình cơ bản
        except Exception as e:
            print(f"Không thể tải checkpoint: {e}")
            print("Sử dụng mô hình PhoBERT cơ bản")
        
        # Đặt mô hình ở chế độ đánh giá
        model.eval()
        return True
    except Exception as e:
        print(f"Lỗi tổng thể khi khởi tạo mô hình: {e}")
        return False

def get_model():
    global model, tokenizer
    if model is None:
        success = init_model()
        if not success:
            raise Exception("Không thể khởi tạo mô hình")
    return model, tokenizer

def predict(text):
    if not text:
        return None
    
    # Lấy mô hình khi cần
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

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok", 
        "message": "Ứng dụng đang hoạt động",
        "model_status": "Đã tải checkpoint" if model_loaded else "Sử dụng mô hình cơ bản"
    })

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
            "model_type": "Đã huấn luyện" if model_loaded else "Cơ bản"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "data": None
        }), 500

# Khởi tạo mô hình khi khởi động
try:
    init_model()
except Exception as e:
    print(f"Lỗi khi khởi tạo mô hình: {e}")
    print("Ứng dụng sẽ tải mô hình khi có request đầu tiên")

if __name__ == "__main__":
    if not init_model():
        print("Cảnh báo: Mô hình chưa được khởi tạo")
    app.run(host="0.0.0.0", port=5000, debug=False) 