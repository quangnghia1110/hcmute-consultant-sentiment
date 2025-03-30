import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import gdown
import zipfile

# Thiết lập tiêu đề trang
st.set_page_config(page_title="Phân tích cảm xúc Tiếng Việt", layout="wide")

# Thêm đoạn code phía trên phần tabs để theo dõi tab hiện tại
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

def handle_tab_change(tab_index):
    st.session_state.current_tab = tab_index

def download_and_extract(file_id, output_dir='checkpoints'):
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Tạo URL download từ file id Google Drive
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Tải file zip về
    zip_path = 'checkpoints.zip'
    gdown.download(url, zip_path, quiet=False)
    
    # Giải nén file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Xóa file zip tạm
    os.remove(zip_path)
    
    print(f'Đã tải và giải nén file vào thư mục {output_dir}')

# Tải model và tokenizer
@st.cache_resource
def load_model():
    # Thông tin file checkpoint
    checkpoint_filename = "phobert-sentiment-epoch=08-val_f1=0.9431.ckpt"
    checkpoint_dir = "checkpoints"
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_filename}"
    
    # Tạo thư mục checkpoints nếu chưa tồn tại
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Kiểm tra nếu file checkpoint chưa tồn tại, tải xuống từ Google Drive
    if not os.path.exists(checkpoint_path):
        with st.spinner("Đang tải model từ Google Drive..."):
            # ID của file trên Google Drive
            file_id = "1EGhZW09uCNWE7HDyOIFFoNGzPEsAINS9"
            
            try:
                # Tải và giải nén file từ Google Drive
                download_and_extract(file_id, checkpoint_dir)
                st.success("Đã tải model thành công!")
            except Exception as e:
                st.error(f"Lỗi khi tải model: {str(e)}")
                raise
    else:
        # Hiển thị thông báo khi sử dụng file đã tồn tại
        st.info(f"Sử dụng model đã tồn tại tại {checkpoint_path}")
    
    # Tải tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    # Tải model từ checkpoint đã lưu
    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
    # Loại bỏ prefix 'model.' nếu có
    if all(k.startswith('model.') for k in state_dict['state_dict'].keys()):
        state_dict['state_dict'] = {k[6:]: v for k, v in state_dict['state_dict'].items() 
                                if k.startswith('model.')}
    model.load_state_dict(state_dict['state_dict'])

    # Đưa model về chế độ đánh giá
    model.eval()
    
    return model, tokenizer

# Tải model và tokenizer
model, tokenizer = load_model()

# Mapping kết quả
id2label = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}

# Bổ sung hàm tiền xử lý
def preprocess_text(text):
    """Tiền xử lý văn bản tiếng Việt"""
    if not text:
        return text
    
    # Loại bỏ dấu cách thừa
    text = ' '.join(text.split())
    
    # Loại bỏ một số ký tự đặc biệt không cần thiết
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    return text

# Hàm dự đoán cảm xúc cho một câu
def predict_sentiment(text):
    try:
        # Kiểm tra đầu vào
        if not text or not isinstance(text, str):
            return {
                "text": str(text),
                "sentiment": "Lỗi",
                "confidence": 0.0,
                "probabilities": {id2label[i]: "0.0000" for i in id2label},
                "error": "Văn bản không hợp lệ"
            }
            
        # Tiền xử lý văn bản
        text = preprocess_text(text)
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        
        # Dự đoán
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Lấy xác suất cho mỗi nhãn
        probs = probabilities[0].tolist()
        
        # Logic thông minh hơn: nếu độ tin cậy thấp giữa các nhãn, xem xét lại
        max_prob = max(probs)
        second_max_prob = sorted(probs, reverse=True)[1]
        
        # Nếu sự khác biệt giữa 2 nhãn cao nhất rất nhỏ và nhãn thứ 2 là trung tính
        if max_prob - second_max_prob < 0.1 and probs.index(second_max_prob) == 1:
            predicted_class = 1  # Chọn nhãn trung tính trong trường hợp không chắc chắn
        
        # Kết quả
        result = {
            "text": text,
            "sentiment": id2label[predicted_class],
            "confidence": probs[predicted_class],
            "probabilities": {id2label[i]: f"{prob:.4f}" for i, prob in enumerate(probs)}
        }
        
        return result
    except Exception as e:
        return {
            "text": text,
            "sentiment": "Lỗi",
            "confidence": 0.0,
            "probabilities": {id2label[i]: "0.0000" for i in id2label},
            "error": str(e)
        }

# Tính toán các metrics đánh giá
def calculate_metrics(results, true_labels):
    if not results or not true_labels:
        return None, None, None
    
    # Tính độ chính xác
    correct = 0
    confusion_matrix = {
        0: {0: 0, 1: 0, 2: 0},
        1: {0: 0, 1: 0, 2: 0},
        2: {0: 0, 1: 0, 2: 0}
    }
    
    for result, true_label in zip(results, true_labels):
        if result["sentiment"] != "Lỗi":
            predicted_label = list(id2label.keys())[list(id2label.values()).index(result['sentiment'])]
            confusion_matrix[true_label][predicted_label] += 1
            if predicted_label == true_label:
                correct += 1
    
    accuracy = correct / len(true_labels) if true_labels else 0
    
    # Tính precision, recall, F1 cho từng lớp
    metrics = {}
    for cls in [0, 1, 2]:
        tp = confusion_matrix[cls][cls]
        fp = sum(confusion_matrix[other][cls] for other in [0, 1, 2] if other != cls)
        fn = sum(confusion_matrix[cls][other] for other in [0, 1, 2] if other != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    metrics_df = pd.DataFrame({
        "Nhãn": ["Tiêu cực", "Trung tính", "Tích cực"],
        "Precision": [metrics[0]["precision"], metrics[1]["precision"], metrics[2]["precision"]],
        "Recall": [metrics[0]["recall"], metrics[1]["recall"], metrics[2]["recall"]],
        "F1": [metrics[0]["f1"], metrics[1]["f1"], metrics[2]["f1"]]
    })
    
    cm_df = pd.DataFrame({
        "Dự đoán/Thực tế": ["Tiêu cực", "Trung tính", "Tích cực"],
        "Tiêu cực": [confusion_matrix[0][0], confusion_matrix[1][0], confusion_matrix[2][0]],
        "Trung tính": [confusion_matrix[0][1], confusion_matrix[1][1], confusion_matrix[2][1]],
        "Tích cực": [confusion_matrix[0][2], confusion_matrix[1][2], confusion_matrix[2][2]]
    })
    
    return accuracy, metrics_df, cm_df

# Thêm phần này trước khi khai báo tabs
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

tab_labels = ["Phân tích đơn lẻ", "Phân tích & Đánh giá"]
selected_tab = st.radio("", tab_labels, index=st.session_state.active_tab, horizontal=True, label_visibility="collapsed")
st.session_state.active_tab = tab_labels.index(selected_tab)

# Sau đó, thay vì sử dụng st.tabs(), dùng điều kiện
if st.session_state.active_tab == 0:
    st.header("Phân tích cảm xúc cho một câu")
    text_input = st.text_area("Nhập văn bản cần phân tích:", height=150)
    if st.button("Phân tích", key="single_analysis"):
        if text_input:
            with st.spinner("Đang phân tích..."):
                result = predict_sentiment(text_input)
                
                # Hiển thị kết quả
                sentiment_color = {
                    "Tích cực": "green",
                    "Trung tính": "blue",
                    "Tiêu cực": "red",
                    "Lỗi": "gray"
                }
                
                st.markdown(f"### Kết quả: <span style='color:{sentiment_color[result['sentiment']]}'>{result['sentiment']}</span>", unsafe_allow_html=True)
                
                if result['sentiment'] != "Lỗi":
                    st.markdown(f"**Độ tin cậy:** {result['confidence']:.4f}")
                    
                    # Vẽ biểu đồ xác suất
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(result['probabilities'].keys()),
                            y=[float(val) for val in result['probabilities'].values()],
                            marker_color=['red', 'blue', 'green']
                        )
                    ])
                    fig.update_layout(title="Xác suất cho mỗi nhãn")
                    st.plotly_chart(fig)
                else:
                    st.error(f"Lỗi: {result.get('error', 'Không xác định')}")
        else:
            st.warning("Vui lòng nhập văn bản để phân tích!")

elif st.session_state.active_tab == 1:
    st.header("Phân tích hàng loạt & Đánh giá")
    uploaded_file = st.file_uploader("Tải lên tệp CSV (để phân tích và đánh giá):", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Đọc file CSV
            df = pd.read_csv(uploaded_file)
            
            # Tự động xác định cột mà không hiện thông báo
            has_labels = False
            
            # Mặc định sử dụng cột "text" nếu có
            text_col = "text" if "text" in df.columns else df.columns[0]
            
            # Tự động xác định cột nhãn
            if "label" in df.columns:
                label_col = "label"
                has_labels = True
            elif len(df.columns) > 1:
                # Kiểm tra xem cột thứ hai có phải là cột nhãn không
                second_col = df.columns[1]
                if df[second_col].dtype in ['int64', 'float64'] and df[second_col].isin([0, 1, 2]).all():
                    label_col = second_col
                    has_labels = True
            
            # Phần xử lý tùy theo số lượng cột
            if len(df.columns) == 1:
                # Tự động bật tạo nhãn mà không hiển thị lên giao diện
                st.session_state['create_auto_labels'] = True
            
            texts = df[text_col].tolist()
            labels = df[label_col].tolist() if has_labels and label_col else None
            
            # Xử lý tự động mà không cần nút Phân tích
            results = []
            progress_bar = st.progress(0)
            
            for i, text in enumerate(texts):
                if text:
                    result = predict_sentiment(text)
                    results.append(result)
                progress_bar.progress((i + 1) / len(texts))
            
            # Hiển thị kết quả dưới dạng DataFrame
            results_df = pd.DataFrame([
                {
                    "Văn bản": r["text"], 
                    "Cảm xúc": r["sentiment"], 
                    "Độ tin cậy": r["confidence"] if r["sentiment"] != "Lỗi" else 0,
                    "Tích cực": r["probabilities"].get("Tích cực", "0.0000"),
                    "Trung tính": r["probabilities"].get("Trung tính", "0.0000"),
                    "Tiêu cực": r["probabilities"].get("Tiêu cực", "0.0000")
                } for r in results
            ])
            
            # Tạo nhãn tự động nếu được chọn (chỉ khi file chỉ có 1 cột)
            auto_labels = None
            if len(df.columns) == 1 and st.session_state.get('create_auto_labels', False):
                auto_labels = []
                for result in results:
                    if result["sentiment"] != "Lỗi":
                        predicted_label = list(id2label.keys())[list(id2label.values()).index(result['sentiment'])]
                        auto_labels.append(predicted_label)
                    else:
                        auto_labels.append(None)
            
            # Nếu có nhãn thực tế, thêm vào DataFrame
            if has_labels and labels:
                results_df["Nhãn thật"] = labels
                # Thêm cột phân biệt dự đoán đúng/sai
                results_df["Dự đoán đúng"] = results_df.apply(
                    lambda row: row["Cảm xúc"] == id2label.get(int(row["Nhãn thật"]), "Không xác định") 
                    if row["Cảm xúc"] != "Lỗi" else False, axis=1
                )
            
            # Thêm các cột khác từ dataframe gốc nếu có
            if len(df.columns) > 1:
                for col in df.columns:
                    if col != text_col and (not has_labels or col != label_col) and col not in results_df.columns:
                        results_df[col] = df[col].tolist()
            
            st.dataframe(results_df, use_container_width=True, height=500)
            
            # Tạo CSV để tải xuống
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Tải kết quả dưới dạng CSV",
                csv,
                "results.csv",
                "text/csv",
                key='download-csv'
            )
            
        except Exception as e:
            st.error(f"Lỗi khi xử lý tệp: {str(e)}")

# Thêm thông tin footer
st.markdown("---")
st.markdown("### Thông tin")
st.markdown("""
- Mô hình: PhoBERT (vinai/phobert-base)
- Checkpoint: phobert-sentiment-epoch=08-val_f1=0.9431.ckpt
- Hỗ trợ phân loại cảm xúc tiếng Việt: Tiêu cực, Trung tính, Tích cực
- Thực hiện và phát triển: Ngô Quang Nghĩa
- Email liên hệ: nqndev.work@gmail.com
""") 