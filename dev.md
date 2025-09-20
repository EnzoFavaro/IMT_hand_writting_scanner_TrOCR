# Projeto: OCR de Redações ENEM com TrOCR

## Objetivo

Construir um serviço em Python capaz de:

1. Receber imagens de redações manuscritas em português-BR.
2. Pré-processar as imagens para melhorar a qualidade.
3. Usar **TrOCR** para reconhecer o texto.
4. Retornar texto transcrito e nível de confiança.

Correções ortográficas/gramaticais não serão feitas aqui — isso ficará em outro serviço.

---

## Padrões de desenvolvimento

* **Linguagem:** Python 3.11+
* **Framework API:** FastAPI
* **Inferência:** HuggingFace TrOCR (torch) + opção ONNXRuntime para CPU
* **Pré-processamento:** OpenCV + Pillow
* **Execução local:** Uvicorn (`uvicorn app.main:app --reload`)
* **Armazenamento:** somente local (imagens processadas e saídas em pastas do projeto)
* **Testes:** Pytest

---

## Estrutura de pastas

```
tcc-enem/
├─ app/
│  ├─ main.py          # FastAPI app
│  ├─ ocr/
│  │  ├─ trocr_infer.py
│  │  ├─ model_load.py
│  ├─ preprocess/
│  │  ├─ image_utils.py
│  ├─ schemas.py       # modelos de request/response
│  ├─ tests/
├─ models/             # modelos baixados ou convertidos
├─ data/               # entrada/saída local
├─ requirements.txt
```

---

## API

**POST /analyze**

* Entrada: imagem (PNG/JPEG/PDF página única).
* Saída JSON:

  ```json
  {
    "ocr_text": "texto transcrito",
    "ocr_by_line": [
      {"line_idx": 0, "text": "linha reconhecida", "confidence": 0.92},
      {"line_idx": 1, "text": "segunda linha", "confidence": 0.88}
    ]
  }
  ```

**GET /health**

* Verifica se API e modelo estão disponíveis.

---

## Pré-processamento

1. Converter para grayscale.
2. Corrigir rotação (deskew simples).
3. Aplicar redução de ruído leve.
4. Ajustar contraste e binarização adaptativa.
5. Redimensionar para entrada do modelo.

---

## OCR (TrOCR)

* Usar `microsoft/trocr-small-handwritten` como modelo base.
* Suporte a CPU.
* (Opcional) Exportar para ONNX e quantizar para acelerar.
* Processar por linha quando possível.

---

## Desenvolvimento — passos

1. Criar FastAPI app com endpoint `/analyze`.
2. Implementar carregamento do modelo TrOCR.
3. Implementar pré-processamento mínimo (resize + grayscale).
4. Integrar OCR linha a linha.
5. Retornar texto com confiança.
6. Testar com imagens locais e armazenar resultados em `data/`.

---

## Bibliotecas

* fastapi
* uvicorn
* opencv-python
* pillow
* torch
* transformers
* onnxruntime (opcional)
* pytest

---
