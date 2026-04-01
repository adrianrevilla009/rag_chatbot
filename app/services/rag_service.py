"""
PIPELINE RAG COMPLETO
======================
Este es el fichero más importante del proyecto. Aquí se orquesta el patrón RAG:

  1. RETRIEVE: Buscar los chunks más relevantes para la pregunta
  2. AUGMENT:  Construir el prompt con esos chunks como contexto
  3. GENERATE: Enviar el prompt al LLM y obtener la respuesta

Sin RAG: LLM responde con su conocimiento de entrenamiento (puede alucinar
         sobre documentos que no conoce).

Con RAG: LLM recibe el contexto real de los documentos y solo tiene que
         sintetizar la respuesta. Mucho más fiable y actualizable.

FLUJO COMPLETO:
    pregunta del usuario
         ↓
    embedding de la pregunta
         ↓
    búsqueda vectorial en pgvector → top 5 chunks más similares
         ↓
    construcción del prompt con los chunks
         ↓
    llamada a Groq (LLaMA 3.3 70B)
         ↓
    respuesta + fuentes citadas
"""

import uuid
from typing import AsyncGenerator

from groq import AsyncGroq
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger
from app.services.cache_service import get_cache_service
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store

logger = get_logger(__name__)

# El system prompt define el comportamiento del LLM
# En producción esto lo tendrías en base de datos para poder cambiarlo sin deploy
SYSTEM_PROMPT = """Eres un asistente experto en análisis de documentos.
Los documentos pueden estar en cualquier idioma. Responde siempre en el idioma de la pregunta.
Tu función es responder preguntas basándote EXCLUSIVAMENTE en el contexto proporcionado.

Reglas:
1. Si la respuesta está en el contexto, responde de forma clara y precisa.
2. Si la información NO está en el contexto, di explícitamente: "No encuentro información sobre eso en los documentos disponibles."
3. NUNCA inventes información que no esté en el contexto.
4. Cita siempre el documento y página de donde proviene la información.
5. Si hay información contradictoria en distintos documentos, señálalo.
6. Responde en el mismo idioma que la pregunta.

Formato de respuesta:
- Respuesta directa y concisa
- Si aplica, menciona: "Fuente: [nombre del documento], página [X]"
"""


class RAGService:

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.cache_service = get_cache_service()
        self.groq_client = AsyncGroq(api_key=settings.groq_api_key)

    async def chat(
        self,
        db: AsyncSession,
        question: str,
        conversation_history: list[dict] | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> dict:
        """
        Punto de entrada principal del chatbot.

        Parámetros:
        - question: la pregunta del usuario
        - conversation_history: mensajes anteriores (para mantener contexto)
        - document_ids: limitar la búsqueda a documentos específicos

        Retorna:
        {
            "answer": str,           # respuesta del LLM
            "sources": list[dict],   # chunks usados (para citar fuentes)
            "cached": bool,          # si vino de caché
        }
        """
        doc_ids_str = [str(did) for did in document_ids] if document_ids else None

        # 1. Intentar caché primero
        cached = await self.cache_service.get(question, doc_ids_str)
        if cached:
            cached["cached"] = True
            return cached

        # 2. RETRIEVE: Buscar chunks relevantes
        relevant_chunks = await self._retrieve(db, question, document_ids)

        if not relevant_chunks:
            return {
                "answer": "No encuentro información relevante en los documentos disponibles. ¿Puedes reformular tu pregunta o subir documentos relacionados?",
                "sources": [],
                "cached": False,
            }

        # 3. AUGMENT: Construir el contexto para el LLM
        context = self._build_context(relevant_chunks)

        # 4. GENERATE: Llamar al LLM
        answer = await self._generate(question, context, conversation_history)

        response = {
            "answer": answer,
            "sources": [
                {
                    "chunk_id": chunk["id"],
                    "filename": chunk["filename"],
                    "page_number": chunk.get("page_number"),
                    "score": round(chunk["score"], 3),
                    "excerpt": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                }
                for chunk in relevant_chunks
            ],
            "cached": False,
        }

        # 5. Guardar en caché para próximas peticiones iguales
        await self.cache_service.set(question, response, doc_ids_str)

        return response

    async def _retrieve(
        self,
        db: AsyncSession,
        question: str,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[dict]:
        """
        RETRIEVE: Convierte la pregunta en embedding y busca chunks similares.

        ¿Por qué embebemos la pregunta con el mismo modelo que los chunks?
        Porque los embeddings solo son comparables si fueron generados con
        el mismo modelo. Mezclar modelos daría resultados sin sentido.
        """
        logger.info("retrieving_chunks", question=question[:100])

        # Embedding de la pregunta
        query_embedding = self.embedding_service.embed_text(question)

        # Búsqueda vectorial
        chunks = await self.vector_store.similarity_search(
            db=db,
            query_embedding=query_embedding,
            document_ids=document_ids,
        )

        logger.info("chunks_retrieved", count=len(chunks))
        return chunks

    def _build_context(self, chunks: list[dict]) -> str:
        """
        AUGMENT: Construye el contexto que se incluirá en el prompt.

        Ordenamos por chunk_index para que los fragmentos del mismo documento
        aparezcan en orden lógico de lectura.
        """
        # Agrupar por documento para mejor legibilidad en el prompt
        docs: dict[str, list] = {}
        for chunk in chunks:
            filename = chunk["filename"]
            if filename not in docs:
                docs[filename] = []
            docs[filename].append(chunk)

        context_parts = []
        for filename, doc_chunks in docs.items():
            # Ordenar chunks por posición en el documento
            doc_chunks.sort(key=lambda x: x["chunk_index"])

            context_parts.append(f"=== Documento: {filename} ===")
            for chunk in doc_chunks:
                page_info = f" (página {chunk['page_number']})" if chunk.get("page_number") else ""
                context_parts.append(f"[Relevancia: {chunk['score']:.2f}{page_info}]")
                context_parts.append(chunk["content"])
                context_parts.append("")

        return "\n".join(context_parts)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        # Reintenta hasta 3 veces con espera exponencial: 1s, 2s, 4s
        # ¿Por qué? Los LLMs externos pueden fallar por rate limits o timeouts.
        # En producción SIEMPRE tienes reintentos en llamadas a servicios externos.
    )
    async def _generate(
        self,
        question: str,
        context: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """
        GENERATE: Llama al LLM con el contexto recuperado.

        Construimos los mensajes siguiendo el formato de chat:
        - system: instrucciones del asistente
        - user/assistant: historial de conversación (para mantener contexto)
        - user: pregunta actual con el contexto inyectado
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Añadir historial de conversación (últimos 6 mensajes = 3 turnos)
        # ¿Por qué limitar? Cada mensaje extra consume tokens y aumenta la latencia.
        # 3 turnos de historial es suficiente para mantener contexto conversacional.
        if conversation_history:
            messages.extend(conversation_history[-6:])

        # El prompt final: contexto + pregunta
        user_message = f"""CONTEXTO DE LOS DOCUMENTOS:
{context}

PREGUNTA: {question}

Responde basándote exclusivamente en el contexto anterior."""

        messages.append({"role": "user", "content": user_message})

        logger.info("calling_groq", model=settings.groq_model, messages=len(messages))

        response = await self.groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            # temperature=0.1: casi determinista. Para RAG queremos respuestas
            # consistentes, no creativas. Alta temperature → más alucinaciones.
        )

        answer = response.choices[0].message.content
        logger.info("groq_response_received", tokens=response.usage.total_tokens)
        return answer


_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
