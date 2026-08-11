from src.state import AgentState
from src.agents.base import get_llm, retrieve_context, format_context_block, sources_from_docs
from config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Tax Education Agent. Explain tax concepts and
account types (401k, Traditional/Roth IRA, HSA, capital gains, tax brackets,
etc.) clearly and accurately, grounded in the provided knowledge-base
context. You are NOT a licensed tax professional — never give a definitive
answer about a user's specific tax liability or filing decisions; explain
the general rules and recommend consulting a CPA or tax advisor for their
specific situation. Cite which concepts come from the knowledge base."""


def run(state: AgentState) -> AgentState:
    query = state["query"]
    try:
        docs = retrieve_context(query, category="tax_education")
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
        logger.error(f"Tax education agent failed: {e}")
        state["final_response"] = "I couldn't answer that tax question right now. Please try again."
        state["error"] = str(e)
    return state
