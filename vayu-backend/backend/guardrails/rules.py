import re
import asyncio

# Zero-latency compiled regex for guardrails
OFF_TOPIC_REGEX = re.compile(r'\b(weather|joke|story|poem|code me|write code|translate|play a song)\b', re.IGNORECASE)
INJECTION_REGEX = re.compile(r'\b(ignore previous|system prompt|you are a|forget everything|bypass)\b', re.IGNORECASE)
CITATION_REGEX = re.compile(r'\[ID:\s*([^\]]+)\]')

class Guardrails:
    @staticmethod
    def check_input(query: str) -> dict:
        """Deterministic, zero-latency input validation."""
        if INJECTION_REGEX.search(query):
            return {"safe": False, "reason": "Prompt injection attempt detected."}
        if OFF_TOPIC_REGEX.search(query):
            return {"safe": False, "reason": "Off-topic query detected."}
        return {"safe": True}

    @staticmethod
    def check_grounding(answer: str, retrieved_parent_ids: set) -> bool:
        """
        Validates that generated citations exist in the retrieved parent IDs.
        Looks for patterns like [ID: 12345] in the answer.
        """
        citations = set(CITATION_REGEX.findall(answer))
        if not citations:
            return True # No citations explicitly made, so we pass it, or we could be strict.
        
        # Check if all citations in the answer are actually from the retrieved context
        return citations.issubset(retrieved_parent_ids)

    @staticmethod
    async def run_parallel_input_guardrail(query: str):
        """Runs input validation asynchronously so it can be parallelized with vector search."""
        return await asyncio.to_thread(Guardrails.check_input, query)
