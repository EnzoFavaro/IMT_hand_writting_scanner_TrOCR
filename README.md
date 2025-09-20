# Projeto: IMT Hand Writting Scanner

## Visão geral
Um serviço simples e robusto para converter redações manuscritas (português-BR) em texto. O sistema roda em CPU e é exposto via API REST para ser consumido por outros serviços (por exemplo, um serviço externo de correção de textos).

### Requisitos principais
- Suportar imagens de redações manuscritas (fotografias e scans).
- Boa taxa de reconhecimento em português do Brasil (melhor possível com TrOCR e fine-tuning se necessário).
- Simplicidade e robustez (tratamento de erros, logs, retries leves).
- Roda em CPU (otimizações: quantização, ONNX, batching, caching).
- Encapsulado em API (documentada) que retorna texto transcrito e nível de confiança.
- Armazenamento apenas local (sem uso de cloud ou storage externo).

---

## Arquitetura em alto nível

```
[Cliente / Scanner / Aplicativo]
        |
   (POST image)
        v
[API REST (FastAPI)]
        |
        +--> Pré-processamento de imagem (OpenCV, Pillow)
        |
        +--> OCR (TrOCR - HuggingFace / ONNX CPU)
        |
        +--> Pós-processamento simples (montagem de linhas, normalização)
        |
        v
   [Resposta JSON: texto transcrito + confiança]
```

Componentes principais:
- **API (FastAPI)**: rota única `/analyze` que orquestra o pipeline.
- **Pré-processamento de imagem**: deskew, crop, contraste, redução de ruído, binarização, redimensionamento, normalização para o modelo.
- **OCR (TrOCR)**: inferência com TrOCR adaptado para manuscrito; preferir modelo menor e quantizado para CPU.
- **Montagem de texto**: combina linhas extraídas e normaliza espaços.
- **Armazenamento local**: manter resultados e logs localmente (arquivos texto e imagens processadas).

---

## Fluxo de dados detalhado
1. Cliente envia imagem (JPEG/PNG/PDF página única) via POST.
2. API valida imagem (tamanho, tipo). Armazena temporariamente em disco local.
3. Pré-processador corrige orientação, remove ruído, aplica binarização adaptativa e normaliza resolução.
4. Segmentação: extrai linhas (opcional). Produz imagens por linha para passar ao OCR.
5. OCR: inferência linha-a-linha com TrOCR.
6. Pós-processamento: junta linhas, corrige espaços.
7. Resposta: texto OCR e nível de confiança por linha/palavra.

---

## Estrutura de pastas (sugestão)
```
tcc-enem/
├─ app/
│  ├─ main.py                # FastAPI app + endpoints
│  ├─ api/
│  │  ├─ endpoints.py        # endpoints principais
│  ├─ core/
│  │  ├─ config.py           # configurações (paths, model)
│  │  ├─ logger.py
│  ├─ ocr/
│  │  ├─ trocr_infer.py      # wrapper de inferência (ONNX/Torch)
│  │  ├─ model_load.py
│  ├─ preprocess/
│  │  ├─ image_utils.py      # deskew, denoise, binarize, resize
│  │  ├─ segmentation.py     # split lines/words (opcional)
│  ├─ schemas.py             # pydantic request/response models
│  ├─ tests/
├─ models/                   # pesos ONNX, vocabulários
├─ data/                     # imagens de teste e saídas locais
├─ notebooks/                # experimentos e visualização
├─ requirements.txt
├─ README.md
```

---

## API — Endpoints sugeridos
**POST /analyze**
- Entrada: multipart/form-data { image: file }
- Saída JSON:
  - `ocr_text` (string)
  - `ocr_by_line` (array de {line_idx, text, confidence})

**POST /ocr**
- Faz só OCR e retorna `ocr_by_line`.

**GET /health**
- Retorna status da API e se o modelo está carregado.

---

## Pré-processamento de imagens (detalhado)
1. **Load & Normalize**: abrir imagem, converter para grayscale.
2. **Orientation/deskew**: detectar rotação e corrigir.
3. **Denoise**: filtro leve para reduzir ruído mantendo traços.
4. **Binarização adaptativa**: melhora contraste.
5. **Resize**: redimensionar mantendo aspecto para o tamanho de entrada do modelo.
6. **Segmentação de linhas**: opcional, caso queira melhorar performance.

---

## OCR — TrOCR (implementação & otimizações CPU)
- Usar implementação HuggingFace (transformers + vision).
- Preferir modelos menores (ex.: `microsoft/trocr-small-handwritten`).
- **Fine-tuning**: se necessário, em manuscritos em português.
- **Conversão ONNX**: exportar para `onnxruntime` com quantização INT8.
- **Batching por linha**: inferir múltiplas linhas por batch.

---

## Dados e fine-tuning
- **Dados manuscritos**: coletar amostras em português (se não houver, criar dataset sintético).
- **Anotações**: texto transcrito ground-truth.
- **Split**: treino / validação / teste.
- Avaliar qualidade por CER (Character Error Rate) e WER (Word Error Rate) durante desenvolvimento.

---

## Deployment
- Executar localmente via `uvicorn`.
- Configurações simples via `.env`.
- Logs e resultados armazenados em pastas locais.

---

## Plano de implementação — passos práticos
1. **MVP (2–3 semanas)**
   - Criar API FastAPI básica.
   - Implementar pré-processamento mínimo (resize, grayscale, deskew simples).
   - Integrar TrOCR com HuggingFace em CPU e inferir em uma imagem.
   - Salvar saídas localmente.

2. **Melhorias (2–4 semanas)**
   - Converter modelo para ONNX e quantizar.
   - Implementar segmentação de linhas.
   - Ajustar heurísticas de pós-processamento.

---

## Tecnologias / Bibliotecas sugeridas
- Python 3.11+
- FastAPI (API)
- Uvicorn
- OpenCV, Pillow, NumPy (pré-process)
- Transformers (HuggingFace) / torch (pytorch)
- onnx + onnxruntime (inference otimizada CPU)
- pytest (testes)

---

## Riscos e Considerações
- **Qualidade dos dados**: OCR depende fortemente da qualidade dos manuscritos usados para treino/fine-tuning.
- **Idioma**: modelos pré-treinados podem não capturar vocabulário específico do ENEM.
- **Privacidade**: armazenar localmente com cuidado para não expor redações.

---

