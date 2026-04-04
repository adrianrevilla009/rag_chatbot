"""
AGENTIC RAG — ITERACIÓN 4
==========================
Novedades respecto a v3:
  - Hybrid search: pasa query_text al vector_store para BM25
  - Citas inline: el prompt pide al LLM que cite [N] en su respuesta
  - Las fuentes se numeran y se mandan con su índice al frontend

CITAS INLINE:
    El LLM recibe los chunks numerados [1], [2], [3]... en el contexto
    y el system prompt le indica que cite el número correspondiente en
    su respuesta. El frontend renderiza [1] como un superíndice clicable
    que despliega el fragmento correspondiente.

    Ejemplo de respuesta del LLM:
    "El salario base es de 32.000€ anuales [1], con revisión anual [2]."
"""

import json
import uuid
from collections.abc import AsyncGenerator

from groq import AsyncGroq
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.services.cache_service import get_cache_service
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.web_search_service import get_web_search_service

logger = get_logger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Busca información en los documentos locales que el usuario ha subido. "
                "Usa esta tool PRIMERO cuando la pregunta pueda estar respondida "
                "por los documentos disponibles (CVs, informes, manuales, contratos, etc.). "
                "Devuelve fragmentos relevantes numerados para que puedas citarlos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "La consulta de búsqueda. Debe ser específica y en inglés "
                            "si los documentos están en inglés, para mejor similitud semántica."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Busca información actualizada en internet usando DuckDuckGo. "
                "Usa esta tool cuando: (1) la información no está en los documentos, "
                "(2) necesitas datos actuales (precios, noticias, estadísticas recientes), "
                "(3) el usuario pregunta sobre algo externo a los documentos. "
                "NO uses esta tool si search_documents ya ha dado una respuesta suficiente."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La consulta de búsqueda web. Sé específico.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

AGENT_SYSTEM_PROMPT = """Eres un asistente inteligente con acceso a dos fuentes de información:
1. Documentos locales del usuario (CVs, informes, contratos, etc.)
2. Búsqueda web en tiempo real (DuckDuckGo)

ESTRATEGIA:
- Usa search_documents primero si la pregunta puede estar en los documentos.
- Usa search_web si necesitas información externa o actualizada.
- Puedes usar ambas tools si es necesario.
- Si ninguna tool da información relevante, responde con lo que sabes.

CITAS INLINE — MUY IMPORTANTE:
- Los resultados de search_documents vienen numerados: [1] fragmento, [2] fragmento, etc.
- En tu respuesta DEBES citar el número de la fuente entre corchetes cada vez que uses información de un fragmento.
- Ejemplo correcto: "El salario base es de 32.000€ [1], con revisión anual en enero [2]."
- Ejemplo incorrecto: "El salario base es de 32.000€." (sin cita)
- Para información de búsqueda web, escribe [web] en lugar de un número.
- Si combinas varias fuentes en una frase: "...según el contrato [1][3]..."

FORMATO:
- Responde siempre en el idioma de la pregunta.
- Sé conciso y directo.
- No incluyas una sección de "Referencias" al final — las citas inline son suficientes."""


class AgenticRAGService:

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.cache_service = get_cache_service()
        self.web_search = get_web_search_service()
        self.groq_client = AsyncGroq(api_key=settings.groq_api_key)

    async def chat(
        self,
        db: AsyncSession,
        question: str,
        conversation_history: list[dict] | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> dict:
        doc_ids_str = [str(did) for did in document_ids] if document_ids else None

        cached = await self.cache_service.get(question, doc_ids_str)
        if cached:
            cached["cached"] = True
            return cached

        answer, sources, tools_used = await self._agent_loop(
            db=db,
            question=question,
            conversation_history=conversation_history,
            document_ids=document_ids,
        )

        response = {
            "answer": answer,
            "sources": sources,
            "cached": False,
            "tools_used": tools_used,
        }

        await self.cache_service.set(question, response, doc_ids_str)
        return response

    async def chat_stream(
        self,
        db: AsyncSession,
        question: str,
        conversation_history: list[dict] | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Generator async SSE. Eventos:
          {"type": "status",  "data": "..."}
          {"type": "sources", "data": [...]}   ← fuentes numeradas
          {"type": "token",   "data": "..."}
          {"type": "done",    "data": {...}}
          {"type": "error",   "data": "..."}
        """
        doc_ids_str = [str(did) for did in document_ids] if document_ids else None

        cached = await self.cache_service.get(question, doc_ids_str)
        if cached:
            cached["cached"] = True
            if cached.get("sources"):
                yield {"type": "sources", "data": cached["sources"]}
            for chunk in _split_into_chunks(cached["answer"]):
                yield {"type": "token", "data": chunk}
            yield {"type": "done", "data": {"tools_used": cached.get("tools_used", []), "cached": True}}
            return

        messages, all_sources, tools_used = await self._agent_loop_collect(
            db=db,
            question=question,
            conversation_history=conversation_history,
            document_ids=document_ids,
        )

        # Solo emitir fuentes si realmente se usó search_documents
        # y el reranker dejó pasar algún chunk relevante
        if all_sources and "search_documents" in tools_used:
            yield {"type": "sources", "data": all_sources}

        full_answer = ""
        async for token in self._stream_final_answer(messages):
            full_answer += token
            yield {"type": "token", "data": token}

        response = {
            "answer": full_answer,
            "sources": all_sources,
            "cached": False,
            "tools_used": tools_used,
        }
        await self.cache_service.set(question, response, doc_ids_str)
        yield {"type": "done", "data": {"tools_used": tools_used, "cached": False}}

    # ─── Agent loop ───────────────────────────────────────────────────────────

    async def _agent_loop(
        self,
        db: AsyncSession,
        question: str,
        conversation_history: list[dict] | None,
        document_ids: list[uuid.UUID] | None,
    ) -> tuple[str, list[dict], list[str]]:
        messages, all_sources, tools_used = await self._agent_loop_collect(
            db=db,
            question=question,
            conversation_history=conversation_history,
            document_ids=document_ids,
        )
        final = await self._call_groq_with_tools(messages, use_tools=False)
        return final.choices[0].message.content or "", all_sources, tools_used

    async def _agent_loop_collect(
        self,
        db: AsyncSession,
        question: str,
        conversation_history: list[dict] | None,
        document_ids: list[uuid.UUID] | None,
    ) -> tuple[list[dict], list[dict], list[str]]:
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history[-6:])

        messages.append({"role": "user", "content": question})

        all_sources: list[dict] = []
        tools_used: list[str] = []
        max_iterations = 3

        for iteration in range(max_iterations):
            logger.info("agent_iteration", iteration=iteration + 1)

            response = await self._call_groq_with_tools(messages)
            msg = response.choices[0].message

            if not msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({"_direct_answer": True, "_content": msg.content or ""})
                return messages, all_sources, tools_used

            assistant_msg = msg.model_dump(exclude_none=True)
            if "content" not in assistant_msg:
                assistant_msg["content"] = ""
            messages.append(assistant_msg)

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                logger.info("executing_tool", tool=tool_name, args=tool_args)
                tools_used.append(tool_name)

                if tool_name == "search_documents":
                    result, sources = await self._tool_search_documents(
                        db=db,
                        query=tool_args["query"],
                        original_question=question,
                        document_ids=document_ids,
                        source_offset=len(all_sources),
                    )
                    all_sources.extend(sources)
                elif tool_name == "search_web":
                    result = await self._tool_search_web(query=tool_args["query"])
                else:
                    result = f"Tool desconocida: {tool_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result,
                })

        logger.warning("agent_max_iterations_reached")
        messages.append({
            "role": "user",
            "content": "Sintetiza la respuesta con la información recopilada, citando las fuentes [N]."
        })
        return messages, all_sources, tools_used

    # ─── Tools ────────────────────────────────────────────────────────────────

    async def _tool_search_documents(
        self,
        db: AsyncSession,
        query: str,
        original_question: str,
        document_ids: list[uuid.UUID] | None,
        source_offset: int = 0,
    ) -> tuple[str, list[dict]]:
        """
        Hybrid search + reranking, con chunks numerados para citas inline.

        source_offset: si ya hay fuentes de llamadas anteriores, los índices
        continúan desde donde se dejaron (evitar [1] duplicados).
        """
        query_embedding = self.embedding_service.embed_text(query)
        chunks = await self.vector_store.similarity_search(
            db=db,
            query_embedding=query_embedding,
            query_text=original_question,   # ← para BM25 y reranker
            document_ids=document_ids,
        )

        if not chunks:
            return "No se encontró información relevante en los documentos locales.", []

        # Construir el contexto con números de cita [N]
        parts = ["=== Resultados en documentos locales ==="]
        parts.append("IMPORTANTE: Cita el número [N] en tu respuesta cuando uses este fragmento.\n")

        sources = []
        for i, chunk in enumerate(chunks, start=source_offset + 1):
            page = f" (pág. {chunk['page_number']})" if chunk.get("page_number") else ""
            method = chunk.get("search_method", "vector")
            rerank_score = chunk.get("rerank_score")

            parts.append(f"[{i}] {chunk['filename']}{page} | método: {method}"
                         + (f" | rerank: {rerank_score:.2f}" if rerank_score is not None else ""))
            parts.append(chunk["content"])
            parts.append("")  # línea en blanco entre chunks

            sources.append({
                "citation_index": i,
                "chunk_id": chunk["id"],
                "filename": chunk["filename"],
                "page_number": chunk.get("page_number"),
                "score": round(chunk["score"], 3),
                "rerank_score": round(rerank_score, 3) if rerank_score is not None else None,
                "search_method": method,
                "excerpt": chunk["content"][:300] + "..." if len(chunk["content"]) > 300 else chunk["content"],
            })

        return "\n".join(parts), sources

    async def _tool_search_web(self, query: str) -> str:
        results = await self.web_search.search(query)
        return self.web_search.format_for_llm(results, query)

    async def _call_groq_with_tools(
        self,
        messages: list[dict],
        use_tools: bool = True,
    ):
        if use_tools:
            clean_messages = [m for m in messages if not (isinstance(m, dict) and "_direct_answer" in m)]
        else:
            # Para la síntesis final sin tools, limpiar tool_calls del historial
            clean_messages = _build_synthesis_messages(messages)

        kwargs = {
            "model": "openai/gpt-oss-120b",
            "messages": clean_messages,
            "max_tokens": 1024,
            "temperature": 0.2,
        }

        if use_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"

        response = await self.groq_client.chat.completions.create(**kwargs)
        logger.info(
            "groq_response",
            tokens=response.usage.total_tokens,
            tool_calls=len(response.choices[0].message.tool_calls or []),
        )
        return response

    async def _stream_final_answer(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        if messages and isinstance(messages[-1], dict) and messages[-1].get("_direct_answer"):
            content = messages[-1].get("_content", "")
            for chunk in _split_into_chunks(content):
                yield chunk
            return

        synthesis_messages = _build_synthesis_messages(messages)

        stream = await self.groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=synthesis_messages,
            max_tokens=1024,
            temperature=0.2,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


def _split_into_chunks(text: str, size: int = 8) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]


def _build_synthesis_messages(messages: list[dict]) -> list[dict]:
    """
    Prepara el historial para la llamada de síntesis final (sin tools).

    El problema: cuando el agente ha usado tools, el historial contiene:
      - Mensajes role=assistant con tool_calls (el LLM pidiendo una tool)
      - Mensajes role=tool con los resultados

    Si enviamos eso a Groq con tool_choice=none o sin tools definidas,
    Groq rechaza la llamada porque ve tool_calls en el historial pero
    no hay tools disponibles para ejecutarlas.

    Solución: reconstruir el historial convirtiendo los bloques
    tool_call + tool_result en un único mensaje role=assistant con
    el contenido de los resultados incrustado como texto. Así el LLM
    tiene toda la información pero sin la estructura de tool_calls.
    """
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        # Saltar marcadores internos
        if isinstance(msg, dict) and "_direct_answer" in msg:
            i += 1
            continue

        # Mensaje de assistant con tool_calls → convertir en texto
        if (isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and msg.get("tool_calls")):

            # Recoger los resultados de todas las tools llamadas
            tool_results = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tool_results.append(
                    f"[Resultado de {messages[j].get('name', 'tool')}]:\n{messages[j].get('content', '')}"
                )
                j += 1

            # Sustituir el bloque entero por un mensaje de contexto limpio
            combined = "\n\n".join(tool_results)
            result.append({
                "role": "assistant",
                "content": f"He consultado las fuentes y obtenido la siguiente información:\n\n{combined}",
            })
            i = j  # saltar los mensajes role=tool que ya procesamos
            continue

        # Mensajes role=tool sueltos (sin assistant previo) → descartar
        if isinstance(msg, dict) and msg.get("role") == "tool":
            i += 1
            continue

        result.append(msg)
        i += 1

    return result


_rag_service: AgenticRAGService | None = None


def get_rag_service() -> AgenticRAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = AgenticRAGService()
    return _rag_service