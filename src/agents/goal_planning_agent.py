from src.state import AgentState
from src.agents.base import get_llm, retrieve_context, format_context_block, sources_from_docs
from config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Goal Planning Agent. Help users think through
financial goals (retirement, home down payment, education, emergency fund,
etc.). Where the user gives numbers (current savings, monthly contribution,
timeline, target amount), show the arithmetic step by step using standard
formulas (future value of a series, the 4% rule, etc.) so the user can
follow the logic. Where inputs are missing, state the assumption you're
using explicitly. This is educational planning support, not personalized
financial advice — note that briefly. Ground general guidance in the
provided knowledge-base context where relevant."""


def run(state: AgentState) -> AgentState:
    query = state["query"]
    try:
        docs = retrieve_context(query, category="goal_planning")
        context = format_context_block(docs)
        llm = get_llm()
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Knowledge base context:\n{context}\n\nUser goal/question: {query}"},
        ])
        state["final_response"] = response.content
        state["retrieved_docs"] = docs
        state["sources"] = sources_from_docs(docs)
    except Exception as e:
        logger.error(f"Goal planning agent failed: {e}")
        state["final_response"] = "I couldn't complete that planning calculation. Please try again."
        state["error"] = str(e)
    return state
