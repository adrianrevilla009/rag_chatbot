"""
PROCESADOR DE DOCUMENTOS
=========================
Este es uno de los componentes más críticos del RAG.
La calidad del chunking determina directamente la calidad de las respuestas.

ESTRATEGIA DE CHUNKING:
Un chunk malo → el retrieval recupera texto irrelevante → el LLM alucina.

Usamos RecursiveCharacterTextSplitter de LangChain porque:
1. Intenta dividir por párrafos primero (\\n\\n)
2. Si el párrafo es demasiado grande, por frases (\\n)
3. Si la frase es demasiado grande, por palabras
4. Nunca corta en medio de una palabra

Esto preserva la coherencia semántica mejor que dividir por número
fijo de caracteres.
"""

import os
import uuid
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentProcessor:

    def __init__(self):
        # ¿Por qué estos separadores en este orden?
        # Intentamos preservar la mayor unidad semántica posible.
        # Solo bajamos al siguiente separador si el chunk sigue siendo muy grande.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",   # 1º: párrafos (mejor separación semántica)
                "\n",     # 2º: saltos de línea
                ". ",     # 3º: frases
                ", ",     # 4º: cláusulas
                " ",      # 5º: palabras
                "",       # Último recurso: caracteres
            ],
        )

    def extract_text(self, file_path: str, file_type: str) -> list[dict]:
        """
        Extrae texto del documento y lo devuelve con metadatos de página.

        Retorna: lista de {"text": str, "page": int}
        Guardamos la página para poder citarla en las respuestas.
        """
        path = Path(file_path)

        if file_type == "pdf":
            return self._extract_pdf(path)
        elif file_type == "txt" or file_type == "md":
            return self._extract_text_file(path)
        elif file_type == "docx":
            return self._extract_docx(path)
        else:
            raise ValueError(f"Tipo de archivo no soportado: {file_type}")

    def _extract_pdf(self, path: Path) -> list[dict]:
        """
        Extrae texto de un PDF con estrategia en dos pasos:
        1. pypdf: rápido, funciona si el PDF tiene texto seleccionable.
        2. OCR con pytesseract: fallback para PDFs escaneados o generados
           desde Canva/Word que embeben el texto como imagen.
        """
        pages = []
        reader = PdfReader(str(path))

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"text": text.strip(), "page": page_num})

        if pages:
            logger.info("pdf_extracted", pages=len(pages), file=path.name, method="pypdf")
            return pages

        # ── Fallback OCR ──────────────────────────────────────────────────
        # El PDF no tiene texto seleccionable — es un PDF de imagen.
        # Convertimos cada página a imagen y aplicamos Tesseract OCR.
        logger.info("pdf_no_text_fallback_ocr", file=path.name)

        try:
            import pytesseract
            from pdf2image import convert_from_path

            # dpi=200: balance calidad/velocidad. 300 es más preciso pero más lento.
            images = convert_from_path(str(path), dpi=200)

            for page_num, image in enumerate(images, start=1):
                # lang="spa+eng": detecta español e inglés simultáneamente
                text = pytesseract.image_to_string(image, lang="spa+eng")
                if text and text.strip():
                    pages.append({"text": text.strip(), "page": page_num})

            logger.info("pdf_extracted", pages=len(pages), file=path.name, method="ocr")

        except Exception as e:
            raise ValueError(f"PDF sin texto seleccionable y OCR falló: {e}")

        return pages

    def _extract_text_file(self, path: Path) -> list[dict]:
        """Para TXT/MD tratamos todo como una sola "página"."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return [{"text": text, "page": 1}]

    def _extract_docx(self, path: Path) -> list[dict]:
        """Extrae texto de documentos Word."""
        from docx import Document
        doc = Document(str(path))
        full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"text": full_text, "page": 1}]

    def create_chunks(self, pages: list[dict]) -> list[dict]:
        """
        Divide el texto en chunks y preserva metadatos.

        El resultado es una lista de chunks listos para embedir e insertar:
        {
            "content": str,         # texto del chunk
            "chunk_index": int,     # posición global
            "page_number": int,     # de qué página viene
            "token_count": int,     # estimación de tokens
        }

        ¿Por qué estimamos tokens con len(text)//4?
        Un token de GPT/LLaMA ≈ 4 caracteres en inglés, ≈ 3-4 en español.
        Es una estimación suficientemente precisa para nuestro uso.
        El tokenizer real requeriría importar tiktoken y es más lento.
        """
        all_chunks = []
        global_index = 0

        for page_data in pages:
            page_text = page_data["text"]
            page_num = page_data["page"]

            # LangChain divide el texto respetando los separadores
            splits = self.splitter.split_text(page_text)

            for split_text in splits:
                if not split_text.strip():
                    continue

                all_chunks.append({
                    "content": split_text.strip(),
                    "chunk_index": global_index,
                    "page_number": page_num,
                    "token_count": len(split_text) // 4,
                })
                global_index += 1

        logger.info("chunks_created", count=len(all_chunks))
        return all_chunks

    def process_file(self, file_path: str, file_type: str) -> list[dict]:
        """Pipeline completo: extrae texto → crea chunks."""
        pages = self.extract_text(file_path, file_type)
        chunks = self.create_chunks(pages)
        return chunks


def get_document_processor() -> DocumentProcessor:
    return DocumentProcessor()
