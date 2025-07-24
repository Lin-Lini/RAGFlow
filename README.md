# RAGFlow
A FastAPI-based Retrieval-Augmented Generation service that ingests PDFs/DOCs/PPTX, segments and indexes them in Elasticsearch using SBERT embeddings, and answers user questions with Vicuna and FLAN-T5, including multilingual translation (ru↔en) via MarianMT

# Сервис обработки текста и вопросов на базе FastAPI

Приложение на **FastAPI**, которое позволяет загружать документы, извлекать и обрабатывать текст, индексировать его в Elasticsearch и отвечать на вопросы пользователей, используя современные модели НЛП (SBERT, Vicuna, FLAN-T5, MarianMT).

## Оглавление

- [Особенности](#особенности)  
- [Требования](#требования)  
- [Установка](#установка)  
- [Конфигурация](#конфигурация)  
- [Запуск приложения](#запуск-приложения)  
- [API Эндпоинты](#api-эндпоинты)  
  - [`POST /upload`](#post-upload)  
  - [`/ask` (WebSocket)](#ask-websocket)  
  - [`POST /process_questions`](#post-process_questions)  
  - [`/prompt` (WebSocket)](#prompt-websocket)  
  - [`POST /vote`](#post-vote)  
- [Используемые модели](#используемые-модели)  
- [Работа с Elasticsearch](#работа-с-elasticsearch)  
- [Замечания по безопасности](#замечания-по-безопасности)  
- [Лицензия](#лицензия)

## Особенности

- **Загрузка файлов**: Поддерживаются форматы PDF, DOC/DOCX, PPTX, TXT.  
- **Извлечение текста**: Автоматический парсинг и очистка текста.  
- **Определение языка и перевод**: Детектирование языка с помощью `langdetect` и перевод через MarianMT (ru↔en).  
- **Сегментация текста**: Разделение на смысловые сегменты с использованием эмбеддингов SBERT.  
- **Извлечение ключевых слов**: TF-IDF + SBERT для определения ключевых слов и их индексации.  
- **Интеграция с Elasticsearch**: Индексация сегментов и ключевых слов, быстрый семантический поиск.  
- **Ответы на вопросы**: Генерация ответов на основе контекста из документов (RAG) и моделями общего знания.  
- **Подсказки в реальном времени**: WebSocket для подсказок ключевых слов.  
- **Сбор обратной связи**: Эндпоинт для голосования (оценки качества ответов).

## Требования

- **Python** 3.10+  
- **PyTorch** с поддержкой CUDA (опционально, но желательно для ускорения модели)  
- **Elasticsearch** 8.x (или совместимая версия)  
- **libmagic** / `file` (для `python-magic`, в Linux: `apt install libmagic1`)  
- Рекомендуемые Python-пакеты (используйте `requirements.txt`):
  ```
  fastapi
  uvicorn[standard]
  python-magic
  nltk
  langdetect
  sentence-transformers
  transformers
  torch
  scikit-learn
  pandas
  xlsxwriter
  elasticsearch
  aiofiles
  pdfminer.six
  python-docx
  python-pptx
  python-dotenv
  ```

## Установка

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject

python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Загрузка данных NLTK

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

(Скрипт приложения может сам это делать при старте, если предусмотрено.)

## Конфигурация

Используйте переменные окружения или `.env` файл:

```env
# .env
ES_URL=https://localhost:9200
ES_USERNAME=elastic
ES_PASSWORD=your_password

MODEL_SBERT_NAME=paraphrase-MiniLM-L6-v2
MODEL_VICUNA_PATH=./llama-base_fine-tuned
MODEL_FLAN_NAME=google/flan-t5-base
MODEL_MARIAN_RU_EN=Helsinki-NLP/opus-mt-ru-en
MODEL_MARIAN_EN_RU=Helsinki-NLP/opus-mt-en-ru

HF_AUTH_TOKEN=your_hf_token   # если модели приватные на HF Hub
```

В коде используйте `os.getenv` или `python-dotenv` для загрузки `.env`.

## Запуск приложения

```bash
uvicorn fastapi_rag_app:app --host 0.0.0.0 --port 8000 --reload
```

> **Примечание:**  
> `fastapi_rag_app:app` — это путь к модулю и объекту приложения FastAPI. Замените на ваш файл/объект при необходимости (например, `main:app`).

### Docker / Docker Compose (опционально)

Создайте `Dockerfile` и `docker-compose.yml`, чтобы поднять сервис и Elasticsearch (и Kibana) одной командой. Это упростит деплой.

## API Эндпоинты

### `POST /upload`

Загружает документ.

- **Формат запроса**: multipart/form-data, поле `file`.
- **Ответ**:
  ```json
  {
    "filename": "document.pdf",
    "checksum": "abc123..."
  }
  ```

**Пример (curl):**
```bash
curl -X POST "http://localhost:8000/upload"   -F "file=@/path/to/your/doc.pdf"
```

### `/ask` WebSocket

Реальное время ответы на вопросы.

- Подключитесь по WebSocket к `ws://localhost:8000/ask`.
- Отправьте строку вопроса.
- Получите JSON с полями:
  - `answer_flan` — ответ модели FLAN-T5
  - `answer` — ответ модели Vicuna
  - `relevance` — список релевантных сегментов (файл, страница, score, текст)

**Пример (JS в браузере):**
```js
const ws = new WebSocket("ws://localhost:8000/ask");
ws.onopen = () => ws.send("Какой у нас бюджет?");
ws.onmessage = (event) => console.log(JSON.parse(event.data));
```

### `POST /process_questions`

Обработка Excel с вопросами:

- **Запрос**: multipart/form-data, поле `file` — Excel с колонкой `question`.
- **Ответ**: Excel-файл с ответами (колонки `answer`, `filename`, `slide_number`).

**Пример (curl):**
```bash
curl -X POST "http://localhost:8000/process_questions"   -F "file=@questions.xlsx"   -o processed_questions.xlsx
```

### `/prompt` WebSocket

Подсказки по ключевым словам.

- Отправьте префикс.
- Вернётся список ключевых слов.

### `POST /vote`

Сохранение оценки ответа.

- **Запрос (JSON)**:
  ```json
  {
    "question": "...",
    "answer": "...",
    "score": 5
  }
  ```
- **Ответ**:
  ```json
  { "status": "success", "data": { ... } }
  ```

## Используемые модели

- **SBERT**: `paraphrase-MiniLM-L6-v2` — извлечение эмбеддингов.
- **Vicuna** (например, `lmsys/vicuna-7b-v1.5`) — генерация развернутых ответов.
- **FLAN-T5** (`google/flan-t5-base`) — seq2seq-задачи.
- **MarianMT** (`Helsinki-NLP/opus-mt-ru-en`, `Helsinki-NLP/opus-mt-en-ru`) — RU↔EN перевод.

## Работа с Elasticsearch

- Индексы создаются автоматически при старте (при наличии соответствующих функций), либо вручную:
  - `uploaded_files`  
  - `keywords`  
  - `text_segments`  
  - `learning` и `learning-score` (для обучения)  

Опишите мэппинги (mapping) в отдельных JSON или используйте уже существующие в коде.

## Замечания по безопасности

- **Учётные данные**: Не храните логин/пароль ES в коде. Используйте переменные окружения или секреты.  
- **SSL**: Включите `verify_certs=True` в Elasticsearch клиенте, при наличии валидного сертификата.  
- **Файлы**: Ограничьте максимальный размер загрузки, проверяйте MIME-типы и расширения.  
- **Логи и аудиты**: Логируйте обращение к API (без утечки персональных данных), мониторьте запросы.  
- **Rate limiting / Auth**: Добавьте авторизацию и ограничения, если сервис публичный.

## Лицензия

Проект распространяется по лицензии [MIT](LICENSE).  
