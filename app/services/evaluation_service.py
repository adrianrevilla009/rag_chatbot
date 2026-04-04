"""
SERVICIO DE EVALUACIÓN RAGAS
==============================
Implementa las 4 métricas clave de RAGAS usando el LLM como juez (LLM-as-a-judge).

POR QUÉ LLM-AS-A-JUDGE EN VEZ DE LA LIBRERÍA ragas:
    La librería ragas requiere OpenAI por defecto y tiene dependencias pesadas.
    Implementar las métricas directamente con Groq nos da:
      - Control total sobre los prompts de evaluación
      - Posibilidad de usar cualquier LLM
      - Sin dependencias extra
      - Los mismos resultados (ragas internamente hace lo mismo)

LAS 4 MÉTRICAS RAGAS:

  1. FAITHFULNESS (0-1)
     Pregunta: ¿Cada afirmación de la respuesta está soportada por los contextos?
     Método: Extraer afirmaciones de la respuesta → verificar cada una contra contextos
     Fórmula: afirmaciones_soportadas / total_afirmaciones
     Problema que detecta: alucinaciones

  2. ANSWER RELEVANCY (0-1)
     Pregunta: ¿La respuesta responde a la pregunta original?
     Método: Pedir al LLM que genere N preguntas que respondería la respuesta →
             calcular similitud coseno con la pregunta original
     Fórmula: mean(cosine_sim(pregunta_original, pregunta_generada_i))
     Problema que detecta: respuestas off-topic o incompletas

  3. CONTEXT PRECISION (0-1)
     Pregunta: ¿Los chunks recuperados son relevantes? (precisión del retrieval)
     Método: Para cada chunk, verificar si fue útil para la respuesta de referencia
     Fórmula: chunks_útiles_en_top_k / k (con penalización por posición)
     Problema que detecta: retrieval ruidoso con chunks irrelevantes

  4. CONTEXT RECALL (0-1)
     Pregunta: ¿Se recuperaron todos los chunks necesarios?
     Método: Verificar qué afirmaciones del ground_truth están cubiertas por contextos
     Fórmula: afirmaciones_ground_truth_cubiertas / total_afirmaciones_ground_truth
     Problema que detecta: retrieval que pierde información relevante
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field

import numpy as np
from groq import AsyncGroq
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embedding_service import get_embedding_service
from app.services.rag_service import get_rag_service

logger = get_logger(__name__)

# Modelo separado para evaluación — puede ser diferente al de producción
# Usamos el mismo para no añadir dependencias, pero en producción podrías
# usar un modelo más capaz (gpt-4o) como juez independiente
EVAL_MODEL = "openai/gpt-oss-120b"


@dataclass
class SampleResult:
    sample_id: str
    question: str
    ground_truth: str
    generated_answer: str = ""
    retrieved_contexts: list[str] = field(default_factory=list)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    error: str | None = None


class EvaluationService:

    def __init__(self):
        self.groq = AsyncGroq(api_key=settings.groq_api_key)
        self.embedding_service = get_embedding_service()
        self.rag_service = get_rag_service()

    # ─── Entry point ──────────────────────────────────────────────────────────

    async def run_evaluation(
        self,
        db: AsyncSession,
        run_id: str,
        samples: list[dict],
    ) -> None:
        """
        Ejecuta la evaluación completa sobre un conjunto de muestras.
        Actualiza la tabla eval_runs y eval_results en tiempo real.
        Se llama en background (Celery o asyncio.create_task).
        """
        logger.info("eval_run_started", run_id=run_id, samples=len(samples))

        results: list[SampleResult] = []

        for sample in samples:
            try:
                result = await self._evaluate_sample(db, sample)
                results.append(result)
                # Guardar resultado parcial inmediatamente
                await self._save_sample_result(db, run_id, result)
                logger.info(
                    "eval_sample_done",
                    sample_id=sample["id"],
                    faithfulness=result.faithfulness,
                    answer_relevancy=result.answer_relevancy,
                )
            except Exception as e:
                logger.error("eval_sample_error", sample_id=sample["id"], error=str(e))
                err_result = SampleResult(
                    sample_id=sample["id"],
                    question=sample["question"],
                    ground_truth=sample["ground_truth"],
                    error=str(e),
                )
                results.append(err_result)
                await self._save_sample_result(db, run_id, err_result)

        # Calcular métricas agregadas
        agg = self._aggregate(results)

        # Actualizar el run con los resultados finales
        await db.execute(
            text("""
                UPDATE eval_runs SET
                    status = 'done',
                    faithfulness = :faithfulness,
                    answer_relevancy = :answer_relevancy,
                    context_precision = :context_precision,
                    context_recall = :context_recall,
                    total_samples = :total_samples,
                    finished_at = NOW()
                WHERE id = :run_id
            """),
            {**agg, "total_samples": len(results), "run_id": run_id},
        )
        await db.commit()
        logger.info("eval_run_done", run_id=run_id, **agg)

    # ─── Evaluate one sample ──────────────────────────────────────────────────

    async def _evaluate_sample(self, db: AsyncSession, sample: dict) -> SampleResult:
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        document_ids = sample.get("document_ids") or None

        # 1. Generar respuesta con el pipeline RAG real
        rag_result = await self.rag_service.chat(
            db=db,
            question=question,
            document_ids=[uuid.UUID(d) for d in document_ids] if document_ids else None,
        )

        generated_answer = rag_result["answer"]
        sources = rag_result.get("sources", [])
        contexts = [s["excerpt"] for s in sources]

        result = SampleResult(
            sample_id=sample["id"],
            question=question,
            ground_truth=ground_truth,
            generated_answer=generated_answer,
            retrieved_contexts=contexts,
        )

        # 2. Calcular las 4 métricas en paralelo
        metrics = await asyncio.gather(
            self._faithfulness(generated_answer, contexts),
            self._answer_relevancy(question, generated_answer),
            self._context_precision(question, ground_truth, contexts),
            self._context_recall(ground_truth, contexts),
            return_exceptions=True,
        )

        result.faithfulness       = metrics[0] if not isinstance(metrics[0], Exception) else None
        result.answer_relevancy   = metrics[1] if not isinstance(metrics[1], Exception) else None
        result.context_precision  = metrics[2] if not isinstance(metrics[2], Exception) else None
        result.context_recall     = metrics[3] if not isinstance(metrics[3], Exception) else None

        return result

    # ─── Métrica 1: Faithfulness ──────────────────────────────────────────────

    async def _faithfulness(self, answer: str, contexts: list[str]) -> float:
        """
        ¿Cada afirmación de la respuesta está soportada por los contextos?

        Paso 1: Extraer afirmaciones atómicas de la respuesta
        Paso 2: Para cada afirmación, verificar si los contextos la soportan
        Score: afirmaciones_soportadas / total_afirmaciones
        """
        if not answer.strip() or not contexts:
            return 0.0

        context_str = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

        # Paso 1: Extraer afirmaciones
        claims_resp = await self._llm_json(f"""
Dado el siguiente texto, extrae todas las afirmaciones atómicas (hechos individuales).
Devuelve SOLO un JSON con la clave "claims" que es una lista de strings.

TEXTO:
{answer}

JSON:""")

        claims = claims_resp.get("claims", [])
        if not claims:
            return 1.0  # Si no hay afirmaciones, no hay nada que verificar

        # Paso 2: Verificar cada afirmación contra los contextos
        verify_resp = await self._llm_json(f"""
Para cada afirmación, indica si está soportada por los contextos proporcionados.
Devuelve SOLO un JSON con la clave "verdicts" que es una lista de 1 (soportada) o 0 (no soportada).
La lista debe tener exactamente {len(claims)} elementos, uno por afirmación.

CONTEXTOS:
{context_str}

AFIRMACIONES:
{json.dumps(claims, ensure_ascii=False)}

JSON:""")

        verdicts = verify_resp.get("verdicts", [])
        if not verdicts:
            return 0.0

        supported = sum(1 for v in verdicts[:len(claims)] if v == 1)
        return round(supported / len(claims), 3)

    # ─── Métrica 2: Answer Relevancy ──────────────────────────────────────────

    async def _answer_relevancy(self, question: str, answer: str) -> float:
        """
        ¿La respuesta responde a la pregunta?

        Genera N preguntas que respondería esta respuesta y calcula
        su similitud coseno con la pregunta original.
        Score alto → la respuesta es directamente relevante a la pregunta.
        """
        if not answer.strip():
            return 0.0

        n_questions = 3

        gen_resp = await self._llm_json(f"""
Dada la siguiente respuesta, genera {n_questions} preguntas diferentes que esta respuesta respondería.
Devuelve SOLO un JSON con la clave "questions" que es una lista de {n_questions} strings.

RESPUESTA:
{answer}

JSON:""")

        generated_questions = gen_resp.get("questions", [])
        if not generated_questions:
            return 0.0

        # Embeddings de la pregunta original y las generadas
        orig_emb = self.embedding_service.embed_text(question)
        orig_arr = np.array(orig_emb)

        similarities = []
        for q in generated_questions[:n_questions]:
            gen_emb = self.embedding_service.embed_text(q)
            gen_arr = np.array(gen_emb)
            # Similitud coseno
            cos_sim = float(np.dot(orig_arr, gen_arr) / (np.linalg.norm(orig_arr) * np.linalg.norm(gen_arr) + 1e-8))
            similarities.append(cos_sim)

        return round(float(np.mean(similarities)), 3)

    # ─── Métrica 3: Context Precision ─────────────────────────────────────────

    async def _context_precision(
        self, question: str, ground_truth: str, contexts: list[str]
    ) -> float:
        """
        ¿Los chunks recuperados son los correctos? (precisión del retrieval)

        Para cada chunk en el top-k, el LLM decide si fue útil para responder
        la pregunta correctamente (comparando con el ground truth).
        Penaliza chunks irrelevantes en posiciones altas (weighted precision).
        """
        if not contexts:
            return 0.0

        verdicts = []
        for ctx in contexts:
            resp = await self._llm_json(f"""
¿El siguiente contexto contiene información útil para responder la pregunta, 
dado que la respuesta correcta es la que se indica?
Devuelve SOLO un JSON con la clave "useful" con valor 1 (útil) o 0 (no útil).

PREGUNTA: {question}
RESPUESTA CORRECTA: {ground_truth}
CONTEXTO: {ctx[:500]}

JSON:""")
            verdicts.append(resp.get("useful", 0))

        if not verdicts:
            return 0.0

        # Average Precision ponderada por posición
        # AP = Σ (precision@k × relevance@k) / total_relevantes
        running_precision = 0.0
        relevant_count = 0
        total_relevant = sum(verdicts)

        if total_relevant == 0:
            return 0.0

        for k, v in enumerate(verdicts, start=1):
            if v == 1:
                relevant_count += 1
                running_precision += relevant_count / k

        return round(running_precision / total_relevant, 3)

    # ─── Métrica 4: Context Recall ────────────────────────────────────────────

    async def _context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        """
        ¿Se recuperaron todos los chunks necesarios?

        Extrae afirmaciones del ground_truth y verifica cuáles están
        cubiertas por los contextos recuperados.
        Score bajo → el retrieval perdió información importante.
        """
        if not contexts or not ground_truth.strip():
            return 0.0

        context_str = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

        # Extraer afirmaciones del ground truth
        claims_resp = await self._llm_json(f"""
Extrae todas las afirmaciones atómicas del siguiente texto de referencia.
Devuelve SOLO un JSON con la clave "claims" que es una lista de strings.

TEXTO:
{ground_truth}

JSON:""")

        claims = claims_resp.get("claims", [])
        if not claims:
            return 1.0

        # Verificar cuáles están cubiertas por los contextos
        verify_resp = await self._llm_json(f"""
Para cada afirmación, indica si está cubierta por alguno de los contextos.
Devuelve SOLO un JSON con la clave "verdicts" que es una lista de 1 (cubierta) o 0 (no cubierta).
La lista debe tener exactamente {len(claims)} elementos.

CONTEXTOS:
{context_str}

AFIRMACIONES:
{json.dumps(claims, ensure_ascii=False)}

JSON:""")

        verdicts = verify_resp.get("verdicts", [])
        if not verdicts:
            return 0.0

        covered = sum(1 for v in verdicts[:len(claims)] if v == 1)
        return round(covered / len(claims), 3)

    # ─── LLM helper ───────────────────────────────────────────────────────────

    async def _llm_json(self, prompt: str) -> dict:
        """Llama al LLM y parsea la respuesta como JSON."""
        try:
            resp = await self.groq.chat.completions.create(
                model=EVAL_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un evaluador de sistemas RAG. Responde SIEMPRE con JSON válido y nada más. Sin texto adicional, sin ```json, solo el JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.0,  # determinístico para evaluación reproducible
            )
            raw = resp.choices[0].message.content or "{}"
            # Limpiar posibles backticks
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(raw)
        except Exception as e:
            logger.warning("eval_llm_json_error", error=str(e))
            return {}

    # ─── Persistence ──────────────────────────────────────────────────────────

    async def _save_sample_result(
        self, db: AsyncSession, run_id: str, result: SampleResult
    ) -> None:
        await db.execute(
            text("""
                INSERT INTO eval_results
                    (run_id, sample_id, generated_answer, retrieved_contexts,
                     faithfulness, answer_relevancy, context_precision, context_recall, error_msg)
                VALUES
                    (:run_id, :sample_id, :generated_answer, :retrieved_contexts,
                     :faithfulness, :answer_relevancy, :context_precision, :context_recall, :error_msg)
            """),
            {
                "run_id": run_id,
                "sample_id": result.sample_id,
                "generated_answer": result.generated_answer,
                "retrieved_contexts": json.dumps(result.retrieved_contexts),
                "faithfulness": result.faithfulness,
                "answer_relevancy": result.answer_relevancy,
                "context_precision": result.context_precision,
                "context_recall": result.context_recall,
                "error_msg": result.error,
            },
        )
        await db.commit()

    # ─── Aggregation ──────────────────────────────────────────────────────────

    def _aggregate(self, results: list[SampleResult]) -> dict:
        def mean_or_none(values):
            clean = [v for v in values if v is not None]
            return round(float(np.mean(clean)), 3) if clean else None

        return {
            "faithfulness":      mean_or_none([r.faithfulness for r in results]),
            "answer_relevancy":  mean_or_none([r.answer_relevancy for r in results]),
            "context_precision": mean_or_none([r.context_precision for r in results]),
            "context_recall":    mean_or_none([r.context_recall for r in results]),
        }


_eval_service: EvaluationService | None = None


def get_evaluation_service() -> EvaluationService:
    global _eval_service
    if _eval_service is None:
        _eval_service = EvaluationService()
    return _eval_service
