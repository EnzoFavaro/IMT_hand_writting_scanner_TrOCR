
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import numpy as np
from spellchecker import SpellChecker
import re

# modelo menor e mais rápido (bom p/ CPU)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")


def preprocess_image(img_path: str):
    # Carrega imagem em escala de cinza
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Binarização adaptativa
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    # Remoção de ruído
    img = cv2.medianBlur(img, 3)
    # Converte para RGB para o PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(img_rgb)
    return pil_img

def postprocess_text(text: str):
    # Remove underscores, caracteres estranhos e espaços extras
    filtered = re.sub(r"_", " ", text)
    filtered = re.sub(r"[^\w\sáéíóúãõâêîôûàèìòùçÁÉÍÓÚÃÕÂÊÎÔÛÀÈÌÒÙÇ]", "", filtered)
    filtered = re.sub(r"\s+", " ", filtered).strip()

    # Tenta separar palavras coladas usando dicionário
    spell = SpellChecker(language='pt')
    # Adicione palavras comuns ao dicionário customizado
    custom_words = ["imagem", "vale", "mais", "que", "mil", "palavras", "uma"]
    spell.word_frequency.load_words(custom_words)

    words = filtered.split()
    unknown = spell.unknown(words)
    corrected = []
    for word in words:
        # Se a palavra for muito longa e desconhecida, tenta dividir
        if word.lower() in unknown and len(word) > 10:
            # Tenta encontrar duas palavras conhecidas dentro da palavra
            for i in range(3, len(word)-3):
                w1, w2 = word[:i], word[i:]
                if w1 in spell and w2 in spell:
                    corrected.extend([w1, w2])
                    break
            else:
                corr = spell.correction(word)
                if corr is None:
                    corr = word
                corrected.append(str(corr))
        elif word.lower() in unknown:
            corr = spell.correction(word)
            if corr is None:
                corr = word
            corrected.append(str(corr))
        else:
            corrected.append(str(word))
    return " ".join(corrected)

def ocr_image(img_path: str):
    image = preprocess_image(img_path)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    generated_ids = model.generate(inputs["pixel_values"])
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    transcription = postprocess_text(transcription)
    return transcription


print(ocr_image("../simple_text.png"))