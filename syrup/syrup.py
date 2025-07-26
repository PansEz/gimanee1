from flask_cors import CORS
from flask import Flask, request, jsonify
import torch
import sentencepiece as spm
from model import GPT, GPTConfig  # โมเดลที่คุณใช้เทรน

app = Flask(__name__)
CORS(app)  # ✅ เปิด CORS ให้ทุก origin

# === โหลด Tokenizer ===
sp = spm.SentencePieceProcessor()
sp.load("thai_health_tokenizer.model")  # ชื่อ tokenizer ที่คุณใช้

# === โหลด Checkpoint และสร้างโมเดล ===
checkpoint = torch.load("ckpt.pt", map_location="cpu", weights_only=True)
model_args = checkpoint["model_args"]
model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint["model"])
model.eval()  # ตั้งเป็นโหมดประเมินผล (inference)

# === ฟังก์ชันสำหรับ encode และ decode ===
def encode(text):
    return sp.encode(text, out_type=int)

def decode(token_ids):
    return sp.decode(token_ids)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        print("📥 รับ request POST /chat")

        data = request.get_json(force=True)
        print("📦 ข้อมูลที่ได้รับ:", data)

        user_input = data.get("message", "")
        print("🟦 user_input:", user_input)

        input_ids = encode(user_input)
        print("🟧 token_ids:", input_ids)

        x = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=50)[0].tolist()
        response_ids = y[len(input_ids):]
        output = decode(response_ids)
        print("output:", output)

        return jsonify({"response": output})

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return jsonify({"response": "ขอโทษค่ะ มีข้อผิดพลาดในการตอบคำถาม"}), 500




# === เริ่มต้นเซิร์ฟเวอร์ ===
if __name__ == "__main__":
    app.run(debug=True)
