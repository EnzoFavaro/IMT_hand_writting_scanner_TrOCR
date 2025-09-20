
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
    # 1. Load & Normalize: abrir imagem, converter para grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    # 3. Denoise: filtro leve para reduzir ruído mantendo traços
    img = cv2.fastNlMeansDenoising(img, None, h=15, templateWindowSize=7, searchWindowSize=21)

    # 4. Binarização adaptativa: melhora contraste
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)

    # 5. Resize: redimensionar mantendo aspecto para o tamanho de entrada do modelo
    # O TrOCR espera 384x384 por padrão
    target_size = (384, 384)
    h, w = img.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Preencher com branco para completar 384x384, sem cortar nada
    canvas = np.ones(target_size, dtype=np.uint8) * 255
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

    # Converte para RGB para o PIL
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(img_rgb)
    # Salva a imagem pré-processada em /results
    import os
    import time
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    timestamp = int(time.time())
    save_path = os.path.join(results_dir, f"{base_name}_processed_{timestamp}.png")
    pil_img.save(save_path)
    return pil_img

def postprocess_text(text: str):
    # Remove underscores, caracteres estranhos e espaços extras
    filtered = re.sub(r"_", " ", text)
    filtered = re.sub(r"[^\w\sáéíóúãõâêîôûàèìòùçÁÉÍÓÚÃÕÂÊÎÔÛÀÈÌÒÙÇ]", "", filtered)
    filtered = re.sub(r"\s+", " ", filtered).strip()

    # Tenta separar palavras coladas usando dicionário
    spell = SpellChecker(language='pt-br')
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
    generated_ids = model.generate(inputs["pixel_values"])
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    transcription = postprocess_text(transcription)
    return transcription


print(ocr_image("../simple_text.png"))