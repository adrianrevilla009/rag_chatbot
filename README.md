# RAG Chatbot — Productivo

Chatbot con patrón RAG (Retrieval-Augmented Generation) usando:
- **pgvector** como vector store
- **Groq** (LLaMA 3.3 70B) como LLM — gratis, sin tarjeta
- **sentence-transformers** para embeddings locales
- **Celery + Redis** para procesamiento asíncrono de documentos
- **FastAPI** como backend async
- **Docker Compose** para orquestar todo

---

## Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado
- Cuenta gratuita en [console.groq.com](https://console.groq.com) (sin tarjeta)

---

## Instalación y arranque

### 1. Clonar y configurar

```bash
# Copia el fichero de variables de entorno
cp .env.example .env
```

Edita `.env` y añade tu Groq API key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

### 2. Levantar todo

```bash
docker compose up --build
```

La primera vez tarda ~5 minutos (descarga imágenes + modelo de embeddings).

### 3. Acceder

| Servicio | URL |
|---|---|
| **Frontend (chatbot)** | http://localhost:3000 |
| **API docs (Swagger)** | http://localhost:8000/docs |
| **Health check** | http://localhost:8000/health/ready |

---

## Uso con curl

### Subir un documento

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@mi_documento.pdf"
```

Respuesta:
```json
{
  "document_id": "uuid-del-documento",
  "job_id": "uuid-del-job",
  "filename": "mi_documento.pdf",
  "message": "Documento recibido. Procesando en background."
}
```

### Consultar estado del procesamiento

```bash
curl http://localhost:8000/documents/jobs/{job_id}
```

Cuando `"status": "done"` y `"progress": 100`, el documento está listo.

### Preguntar

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "¿De qué trata el documento?"}'
```

### Mantener contexto de conversación

```bash
# Primera pregunta — guarda el conversation_id
CONV_ID="uuid-de-la-respuesta"

# Siguiente pregunta en la misma conversación
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"¿Puedes ampliar el último punto?\", \"conversation_id\": \"$CONV_ID\"}"
```

### Filtrar por documento específico

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué dice sobre X?", "document_ids": ["uuid-del-doc"]}'
```

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Nginx :3000)                                 │
│  └── reverse proxy /api/* → FastAPI                     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  FastAPI (:8000)                                        │
│  ├── POST /documents/ingest → encola job en Celery      │
│  ├── POST /chat → pipeline RAG                          │
│  └── GET  /health/ready → liveness check                │
└──────┬──────────────────────┬───────────────────────────┘
       │                      │
┌──────▼──────┐    ┌──────────▼──────────────────────────┐
│   Redis     │    │  Celery Worker                      │
│  (broker +  │◄───│  1. Extrae texto (pypdf/docx)       │
│   cache)    │    │  2. Chunking (RecursiveCharSplitter) │
└─────────────┘    │  3. Embeddings (MiniLM local)       │
                   │  4. INSERT en pgvector               │
                   └──────────────┬──────────────────────┘
                                  │
                   ┌──────────────▼──────────────────────┐
                   │  PostgreSQL + pgvector               │
                   │  ├── documents (metadatos)           │
                   │  ├── chunks (texto + vector 384d)    │
                   │  ├── jobs (estado del procesamiento) │
                   │  └── conversations + messages        │
                   └─────────────────────────────────────┘
```

## Pipeline RAG explicado

```
Pregunta del usuario
       ↓
Embedding de la pregunta (all-MiniLM-L6-v2, local)
       ↓
Búsqueda vectorial en pgvector (HNSW index, similitud coseno)
       ↓
Top-5 chunks más relevantes
       ↓
Construcción del prompt: [system] + [historial] + [contexto] + [pregunta]
       ↓
Groq API → LLaMA 3.3 70B
       ↓
Respuesta + fuentes citadas
       ↓
Guardar en Redis (caché 1h) + PostgreSQL (historial)
```

---

## Parar los servicios

```bash
docker compose down           # para pero conserva datos
docker compose down -v        # para y borra todos los datos
```
