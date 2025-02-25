from langchain_ollama import OllamaLLM
import base64
import requests

# Ollama
llm = OllamaLLM(model="mistral", base_url="http://localhost:11434")

# LLaVA
def summarize_image_with_llava(encoded_image):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llava",  
        "prompt": "Describe the contents of this image.",
        "stream": False,
        "images": [encoded_image]
    }
    response = requests.post(url, json=payload)
    return response.json()["response"]

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# test
image_path = "./images/Swin/figure-1-1.jpg"
encoded_image = encode_image(image_path)
image_summary = summarize_image_with_llava(encoded_image)
print("圖片摘要：", image_summary)