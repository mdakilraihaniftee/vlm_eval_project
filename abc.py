from PIL import Image
img = Image.open("sample.jpg")
model = QwenVLWrapper("Qwen/Qwen2.5-VL-3B-Instruct")
print(model.generate(img, "What is in this picture?"))
