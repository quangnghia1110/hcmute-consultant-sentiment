from fastapi import FastAPI, HTTPException
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from typing import Dict, Any

app = FastAPI(title="Vietnamese Sentiment Analysis API")

# Mapping kết quả
id2label = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}

# Đường dẫn đến file checkpoint
CHECKPOINT_PATH = "checkpoints/phobert-sentiment-epoch=08-val_f1=0.9431.ckpt"

def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
        
        state_dict = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'), weights_only=True)
        if all(k.startswith('model.') for k in state_dict['state_dict'].keys()):
            state_dict['state_dict'] = {k[6:]: v for k, v in state_dict['state_dict'].items() 
                                    if k.startswith('model.')}
        model.load_state_dict(state_dict['state_dict'])
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Khởi tạo model và tokenizer khi start server
model, tokenizer = load_model()

def preprocess_text(text: str) -> str:
    if not text:
        return text
    text = ' '.join(text.split())
    text = text.replace('\n', ' ').replace('\t', ' ')
    return text

def predict_sentiment(text: str) -> Dict[str, Any]:
    start_time = time.time()
    
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text input")
            
        text = preprocess_text(text)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        probs = probabilities[0].tolist()
        max_prob = max(probs)
        second_max_prob = sorted(probs, reverse=True)[1]
        
        if max_prob - second_max_prob < 0.1 and probs.index(second_max_prob) == 1:
            predicted_class = 1
            
        processing_time = time.time() - start_time
        
        return {
            "classify": id2label[predicted_class],
            "time": round(processing_time, 3),
            "percent": {id2label[i]: round(float(prob) * 100, 2) for i, prob in enumerate(probs)}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment")
def analyze_sentiment(text: str):
    try:
        result = predict_sentiment(text)
        return {
            "status": "success",
            "message": "Analysis completed successfully",
            "data": result
        }
    except HTTPException as e:
        return {
            "status": "error",
            "message": str(e.detail),
            "data": None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "data": None
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 