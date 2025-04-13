from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = Flask(__name__)

id2label = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "model/phobert-sentiment-epoch=08-val_f1=0.9431.ckpt")
model = None
tokenizer = None

def init_model():
    global model, tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
        
        state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
        if 'state_dict' in state_dict:
            state_dict = {k.replace('model.', ''): v for k, v in state_dict['state_dict'].items() 
                        if k.startswith('model.')}
            model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise

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
            "data": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "data": None
        }), 500

if __name__ == "__main__":
    init_model()
    app.run(host="0.0.0.0", port=5000, debug=False) 