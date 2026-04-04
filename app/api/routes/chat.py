"""
ENDPOINTS DE CHAT
==================
POST /chat                        → Enviar pregunta, recibir respuesta con fuentes
GET  /chat/conversations          → Listar conversaciones
GET  /chat/conversations/{id}     → Historial de una conversación
DELETE /chat/conversations/{id}   → Eliminar conversación
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
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
    """
    Endpoint principal del chatbot.

    Flujo:
    1. Recuperar o crear conversación
    2. Cargar historial con query explícita (NO lazy load — causa MissingGreenlet en async)
    3. Ejecutar pipeline RAG
    4. Guardar pregunta y respuesta en la DB
    5. Retornar respuesta con fuentes
    """
    # Gestión de conversación
    if request.conversation_id:
        result = await db.execute(
            select(Conversation).where(Conversation.id == request.conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversación no encontrada")
    else:
        conversation = Conversation()
        db.add(conversation)
        await db.flush()

    # Cargar historial con query async explícita
    # NO usar conversation.messages (lazy load) — en SQLAlchemy async el lazy
    # load dispara una query síncrona dentro de un contexto async → MissingGreenlet.
    # Siempre hay que hacer la query explícitamente con await.
    history = []
    if request.conversation_id:
        msgs_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .limit(6)
        )
        recent_msgs = list(reversed(msgs_result.scalars().all()))
        history = [{"role": m.role, "content": m.content} for m in recent_msgs]

    # Ejecutar RAG
    rag = get_rag_service()
    result_data = await rag.chat(
        db=db,
        question=request.question,
        conversation_history=history,
        document_ids=request.document_ids,
    )

    # Guardar ambos mensajes en el historial
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.question,
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
