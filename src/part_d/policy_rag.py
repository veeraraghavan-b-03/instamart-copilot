


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import mock_llm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"

CHUNK_SIZE = 40
OVERLAP = 10


def load_policy_documents():
    documents = {}

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        documents[file_path.name] = file_path.read_text(
            encoding="utf-8"
        )

    return documents


def create_chunks(documents, overlap=OVERLAP):
    chunks = []
    step = CHUNK_SIZE - overlap

    for source, text in documents.items():
        words = text.split()
        start = 0

        while start < len(words):
            end = start + CHUNK_SIZE
            chunk_words = words[start:end]

            chunks.append({
                "source": source,
                "start_word": start,
                "end_word": min(end, len(words)),
                "text": " ".join(chunk_words)
            })

            if end >= len(words):
                break

            start += step

    return chunks


class PolicyRetriever:

    def __init__(self, chunks):
        self.chunks = chunks

        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in chunks]
        )

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.document_vectors
        ).flatten()

        ranked_indices = scores.argsort()[::-1][:k]

        results = []

        for index in ranked_indices:
            chunk = self.chunks[index]

            results.append({
                "source": chunk["source"],
                "score": round(float(scores[index]), 4),
                "text": chunk["text"]
            })

        return results


def answer(question, retriever):

    results = retriever.retrieve(question, k=3)

    # Build the exact context format required by the brief.
    context_parts = []

    for result in results:
        context_parts.append(
            f'[{result["source"]}]\n{result["text"]}'
        )

    context = "\n".join(context_parts)

    user_message = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer the question using only "
                "the supplied policy context. Do not use outside "
                "knowledge or invent information. If the context "
                "does not contain the answer, reply exactly: "
                "I could not find that information in the policy documents."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = mock_llm.chat(messages)

    model_answer = response.choices[0].message.content

    sources = list(dict.fromkeys(
        result["source"] for result in results
    ))

    return {
        "answer": model_answer,
        "sources": sources
    }


def evaluate(retriever, evaluation_data):

    answerable_doc_hits = 0
    answerable_fact_hits = 0
    unanswerable_fallbacks = 0

    answerable_count = 0
    unanswerable_count = 0

    for _, row in evaluation_data.iterrows():

        question = str(row["question"])
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        result = answer(question, retriever)
        model_answer = result["answer"]

        has_expected_doc = (
            pd.notna(expected_doc)
            and str(expected_doc).strip().upper() != "NONE"
        )

        if has_expected_doc:
            answerable_count += 1

            if str(expected_doc) in result["sources"]:
                answerable_doc_hits += 1

            if (
                pd.notna(expected_fact)
                and str(expected_fact) in model_answer
            ):
                answerable_fact_hits += 1

        else:
            unanswerable_count += 1

            if model_answer.startswith("I could not find"):
                unanswerable_fallbacks += 1

    return {
        "answerable_questions": answerable_count,
        "expected_doc_hits": answerable_doc_hits,
        "expected_fact_hits": answerable_fact_hits,
        "unanswerable_questions": unanswerable_count,
        "fallback_hits": unanswerable_fallbacks
    }


def print_evaluation(title, results):

    print("\n" + title)
    print("-" * 55)

    print(
        f'Expected document in top 3: '
        f'{results["expected_doc_hits"]}/'
        f'{results["answerable_questions"]}'
    )

    print(
        f'Answer contains expected fact: '
        f'{results["expected_fact_hits"]}/'
        f'{results["answerable_questions"]}'
    )

    print(
        f'Unanswerable questions with fallback: '
        f'{results["fallback_hits"]}/'
        f'{results["unanswerable_questions"]}'
    )


if __name__ == "__main__":

    documents = load_policy_documents()
    chunks = create_chunks(documents, overlap=10)

    print("Policy documents:", len(documents))
    print("Total chunks:", len(chunks))

    retriever = PolicyRetriever(chunks)

    # Test one question
    test_question = "When can a customer get a refund?"

    print("\nTest question:", test_question)

    test_result = answer(test_question, retriever)

    print("\nAssistant answer:")
    print(test_result["answer"])

    print("\nSources used:")
    print(test_result["sources"])

    # Load the 14 evaluation questions
    evaluation_data = pd.read_csv(EVAL_FILE)

    print("\nEvaluation questions:", len(evaluation_data))

    # Evaluation with 10-word overlap
    overlap_results = evaluate(retriever, evaluation_data)

    print_evaluation(
        "EVALUATION WITH 10-WORD OVERLAP",
        overlap_results
    )

    # Repeat evaluation with zero overlap
    zero_overlap_chunks = create_chunks(
        documents,
        overlap=0
    )

    zero_overlap_retriever = PolicyRetriever(
        zero_overlap_chunks
    )

    zero_results = evaluate(
        zero_overlap_retriever,
        evaluation_data
    )

    print_evaluation(
        "EVALUATION WITH ZERO OVERLAP",
        zero_results
    )
