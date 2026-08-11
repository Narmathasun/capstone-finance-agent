from src.state import AgentState
from src.agents.base import get_llm, retrieve_context, format_context_block, sources_from_docs
from config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Finance Q&A Agent, a specialist in general financial
education for a multi-agent financial assistant. Explain concepts clearly and
accessibly for a non-expert audience. Ground your answer in the provided
knowledge-base context where relevant, and say so when you are drawing on it.
Never give personalized investment, tax, or legal advice — explain concepts
generally and suggest the user consult a licensed professional for their
specific situation. Keep answers concise (3-6 short paragraphs or a bulleted list)."""


def run(state: AgentState) -> AgentState:
    query = state["query"]
    try:
        docs = retrieve_context(query, category="finance_qa")
        context = format_context_block(docs)
        llm = get_llm()
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Knowledge base context:\n{context}\n\nUser question: {query}"},
        ])
        state["final_response"] = response.content
        state["retrieved_docs"] = docs
        state["sources"] = sources_from_docs(docs)
    except Exception as e:
        logger.error(f"Finance QA agent failed: {e}")
        state["final_response"] = (
            "I ran into an issue answering that right now. Could you try rephrasing, "
            "or ask again in a moment?"
        )
        state["error"] = str(e)
    return state
