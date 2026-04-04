"""
AGENTIC RAG — ITERACIÓN 2
==========================
El agente decide qué herramientas usar antes de responder.

FLUJO DEL AGENTE:
    1. LLM recibe pregunta + tools disponibles (JSON Schema)
    2. LLM responde con tool_calls (qué tools quiere usar y con qué args)
    3. Ejecutamos las tools y devolvemos resultados al LLM como role=tool
    4. LLM sintetiza la respuesta final

POR QUÉ NO USAMOS LangChain AgentExecutor:
Implementamos el loop manualmente con la API nativa de Groq.
Así el código es transparente y entiendes exactamente qué ocurre.

NOTA SOBRE GROQ Y TOOL CALLS:
Groq es estricto con el formato del historial cuando hay tool_calls.
El mensaje assistant con tool_calls debe pasarse usando el objeto
nativo de la respuesta (.model_dump()), NO construido a mano como dict.
Construirlo a mano con campos que no coinciden exactamente con lo que
Groq espera provoca BadRequestError.
"""

import json
import uuid

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
                "Devuelve fragmentos relevantes con su puntuación de similitud."
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

FORMATO:
- Responde siempre en el idioma de la pregunta.
- Cita las fuentes: "Según [documento/web]..."
- Sé conciso y directo."""


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

    async def _agent_loop(
        self,
        db: AsyncSession,
        question: str,
        conversation_history: list[dict] | None,
        document_ids: list[uuid.UUID] | None,
    ) -> tuple[str, list[dict], list[str]]:
        """
        Loop agéntico ReAct simplificado.

        CLAVE: para el historial de mensajes con tool_calls, NO construimos
        dicts a mano. Usamos msg.model_dump() del objeto de respuesta de Groq
        para garantizar que el formato es exactamente el que Groq espera.
        Groq valida estrictamente la estructura y rechaza campos incorrectos.
        """
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history[-6:])

        messages.append({"role": "user", "content": question})

        all_sources = []
        tools_used = []
        max_iterations = 3

        for iteration in range(max_iterations):
            logger.info("agent_iteration", iteration=iteration + 1)

            response = await self._call_groq_with_tools(messages)
            msg = response.choices[0].message

            if not msg.tool_calls:
                # El LLM ha decidido responder directamente — fin del loop
                logger.info("agent_finished", iteration=iteration + 1, tools_used=tools_used)
                return msg.content or "", all_sources, tools_used

            # Añadir mensaje del assistant al historial usando model_dump()
            # Esto serializa el objeto Pydantic de Groq con exactamente los campos
            # y el formato que Groq espera en la siguiente llamada.
            # exclude_none=True: evita pasar campos null que Groq rechaza.
            assistant_msg = msg.model_dump(exclude_none=True)
            # Groq espera "content" aunque sea None — pero como string vacío
            if "content" not in assistant_msg:
                assistant_msg["content"] = ""
            messages.append(assistant_msg)

            # Ejecutar cada tool solicitada
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                logger.info("executing_tool", tool=tool_name, args=tool_args)
                tools_used.append(tool_name)

                if tool_name == "search_documents":
                    result, sources = await self._tool_search_documents(
                        db=db,
                        query=tool_args["query"],
                        document_ids=document_ids,
                    )
                    all_sources.extend(sources)
                elif tool_name == "search_web":
                    result = await self._tool_search_web(query=tool_args["query"])
                else:
                    result = f"Tool desconocida: {tool_name}"

                # Mensaje role=tool: resultado de la tool para el LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,   # requerido por Groq junto a tool_call_id
                    "content": result,
                })

        # Agotamos iteraciones — pedir síntesis final sin tools
        logger.warning("agent_max_iterations_reached")
        messages.append({
            "role": "user",
            "content": "Sintetiza la respuesta con la información recopilada."
        })
        final = await self._call_groq_with_tools(messages, use_tools=False)
        return final.choices[0].message.content or "", all_sources, tools_used

    async def _tool_search_documents(
        self,
        db: AsyncSession,
        query: str,
        document_ids: list[uuid.UUID] | None,
    ) -> tuple[str, list[dict]]:
        query_embedding = self.embedding_service.embed_text(query)
        chunks = await self.vector_store.similarity_search(
            db=db,
            query_embedding=query_embedding,
            document_ids=document_ids,
        )

        if not chunks:
            return "No se encontró información relevante en los documentos locales.", []

        docs: dict[str, list] = {}
        for chunk in chunks:
            docs.setdefault(chunk["filename"], []).append(chunk)

        parts = ["=== Resultados en documentos locales ==="]
        for filename, doc_chunks in docs.items():
            doc_chunks.sort(key=lambda x: x["chunk_index"])
            parts.append(f"\n[Documento: {filename}]")
            for chunk in doc_chunks:
                page = f" (pág. {chunk['page_number']})" if chunk.get("page_number") else ""
                parts.append(f"Relevancia: {chunk['score']:.2f}{page}")
                parts.append(chunk["content"])

        sources = [
            {
                "chunk_id": c["id"],
                "filename": c["filename"],
                "page_number": c.get("page_number"),
                "score": round(c["score"], 3),
                "excerpt": c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"],
                "source_type": "document",
            }
            for c in chunks
        ]

        return "\n".join(parts), sources

    async def _tool_search_web(self, query: str) -> str:
        results = await self.web_search.search(query)
        return self.web_search.format_for_llm(results, query)

    async def _call_groq_with_tools(
        self,
        messages: list[dict],
        use_tools: bool = True,
    ):
        kwargs = {
            "model": "openai/gpt-oss-120b",  # Modelo recomendado por Groq para tool calling
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.2,  # Groq recomienda 0.0-0.5 para tool calling
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


_rag_service: AgenticRAGService | None = None


def get_rag_service() -> AgenticRAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = AgenticRAGService()
    return _rag_service