"""
ENDPOINTS DE CHAT
==================
POST   /chat                          → Respuesta completa (no streaming)
GET    /chat/stream                   → SSE streaming (token a token)
GET    /chat/conversations            → Listar conversaciones
GET    /chat/conversations/{id}       → Historial de una conversación
DELETE /chat/conversations/{id}       → Eliminar conversación
"""

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ChatRequest, ChatResponse, ConversationResponse
from app.core.database import get_db
from app.core.logging import get_logger
from app.models.models import Conversation, Message
from app.services.rag_service import get_rag_service

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Endpoint principal del chatbot (respuesta completa, sin streaming)."""
    conversation, history = await _get_or_create_conversation(db, request.conversation_id)

    rag = get_rag_service()
    result_data = await rag.chat(
        db=db,
        question=request.question,
        conversation_history=history,
        document_ids=request.document_ids,
    )

    await _save_messages(db, conversation, request.question, result_data)

    logger.info(
        "chat_completed",
        conversation_id=str(conversation.id),
        cached=result_data["cached"],
        sources=len(result_data["sources"]),
    )

    return ChatResponse(
        answer=result_data["answer"],
        sources=result_data["sources"],
        conversation_id=conversation.id,
        cached=result_data["cached"],
        tools_used=result_data.get("tools_used", []),
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Endpoint de streaming SSE.

    FIX: La conversación se crea y commitea ANTES de abrir el stream.
    Los mensajes se guardan al final con una sesión nueva independiente,
    porque la sesión de FastAPI (db) puede cerrarse durante el streaming.
    """
    conversation, history = await _get_or_create_conversation(db, request.conversation_id)
    conv_id = str(conversation.id)
    # Commit inmediato — la conversación debe existir en DB antes de que
    # el stream empiece a emitir el conversation_id al cliente.
    # Si no commiteamos aquí, el segundo mensaje falla con FK violation
    # porque la conversación no está visible para otras sesiones.
    await db.commit()

    rag = get_rag_service()

    async def event_generator():
        yield _sse({"type": "conversation_id", "data": conv_id})

        full_answer = ""
        final_sources = []
        final_tools = []
        final_cached = False

        try:
            async for event in rag.chat_stream(
                db=db,
                question=request.question,
                conversation_history=history,
                document_ids=request.document_ids,
            ):
                if event["type"] == "sources":
                    final_sources = event["data"]
                    yield _sse(event)
                elif event["type"] == "token":
                    full_answer += event["data"]
                    yield _sse(event)
                elif event["type"] == "done":
                    final_tools = event["data"].get("tools_used", [])
                    final_cached = event["data"].get("cached", False)
                    yield _sse(event)
                elif event["type"] == "error":
                    yield _sse(event)

        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield _sse({"type": "error", "data": str(e)})
            return

        # Guardar mensajes con sesión nueva — la sesión del request (db)
        # puede estar cerrada o en estado inválido al terminar el stream
        if full_answer:
            from app.core.database import AsyncSessionLocal
            async with AsyncSessionLocal() as save_db:
                try:
                    # Recargar la conversación en esta nueva sesión
                    from sqlalchemy import select
                    from app.models.models import Conversation
                    result = await save_db.execute(
                        select(Conversation).where(Conversation.id == conversation.id)
                    )
                    conv = result.scalar_one_or_none()
                    if conv:
                        result_data = {
                            "answer": full_answer,
                            "sources": final_sources,
                            "cached": final_cached,
                            "tools_used": final_tools,
                        }
                        await _save_messages(save_db, conv, request.question, result_data)
                except Exception as e:
                    logger.error("save_messages_error", error=str(e))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # desactiva buffer en nginx
        },
    )


# ─── Conversations ────────────────────────────────────────────────────────────

@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(db: AsyncSession = Depends(get_db)):
    """Lista todas las conversaciones ordenadas por fecha."""
    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .order_by(Conversation.created_at.desc())
        .limit(50)
    )
    conversations = result.scalars().all()
    return [ConversationResponse.model_validate(c) for c in conversations]


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Obtiene el historial completo de una conversación."""
    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")

    return ConversationResponse.model_validate(conversation)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Elimina una conversación y todos sus mensajes."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")

    await db.delete(conversation)
    await db.commit()


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _get_or_create_conversation(
    db: AsyncSession,
    conversation_id: uuid.UUID | None,
) -> tuple[Conversation, list[dict]]:
    """Recupera o crea una conversación y devuelve (conversation, history)."""
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversación no encontrada")
    else:
        conversation = Conversation()
        db.add(conversation)
        await db.flush()

    history = []
    if conversation_id:
        msgs_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .limit(6)
        )
        recent_msgs = list(reversed(msgs_result.scalars().all()))
        history = [{"role": m.role, "content": m.content} for m in recent_msgs]

    return conversation, history


async def _save_messages(
    db: AsyncSession,
    conversation: Conversation,
    question: str,
    result_data: dict,
) -> None:
    """Guarda pregunta y respuesta en la DB."""
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=question,
    )
    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=result_data["answer"],
        sources=result_data["sources"],
    )
    db.add(user_msg)
    db.add(assistant_msg)
    await db.commit()


def _sse(data: dict) -> str:
    """Formatea un dict como evento SSE."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"