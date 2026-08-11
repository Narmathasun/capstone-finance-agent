from unittest.mock import patch, MagicMock
from langgraph.checkpoint.memory import MemorySaver
from src.graph import build_graph, invoke


class TestBuildGraph:
    def test_graph_compiles(self):
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None

    @patch("src.router._router_llm")
    @patch("src.agents.tax_education_agent.get_llm")
    @patch("src.agents.tax_education_agent.retrieve_context")
    def test_end_to_end_routes_to_tax_agent(self, mock_retrieve, mock_agent_llm, mock_router_llm):
        mock_router_llm.invoke.return_value = MagicMock(route="tax_education", confidence=0.95)
        mock_retrieve.return_value = []
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="A Roth IRA is a retirement account.")
        mock_agent_llm.return_value = mock_llm

        graph = build_graph(checkpointer=MemorySaver())
        result = graph.invoke(
            {
                "messages": [],
                "user_id": "u1",
                "session_id": "s1",
                "query": "What is a Roth IRA?",
                "portfolio": [],
            },
            config={"configurable": {"thread_id": "s1"}},
        )
        assert result["route"] == "tax_education"
        assert "Roth IRA" in result["final_response"]

    @patch("src.router._router_llm")
    @patch("src.agents.finance_qa_agent.get_llm")
    @patch("src.agents.finance_qa_agent.retrieve_context")
    def test_default_route_when_router_fails(self, mock_retrieve, mock_agent_llm, mock_router_llm):
        mock_router_llm.invoke.side_effect = Exception("router down")
        mock_retrieve.return_value = []
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="General explanation.")
        mock_agent_llm.return_value = mock_llm

        graph = build_graph(checkpointer=MemorySaver())
        result = graph.invoke(
            {
                "messages": [],
                "user_id": "u1",
                "session_id": "s2",
                "query": "random unrelated question",
                "portfolio": [],
            },
            config={"configurable": {"thread_id": "s2"}},
        )
        assert result["route"] == "finance_qa"


class TestInvokeWrapper:
    @patch("src.graph.get_app")
    def test_invoke_shapes_output(self, mock_get_app):
        mock_app = MagicMock()
        mock_app.invoke.return_value = {
            "final_response": "Answer text",
            "route": "finance_qa",
            "sources": ["kb"],
            "error": None,
        }
        mock_get_app.return_value = mock_app

        result = invoke("What is diversification?", user_id="u1", session_id="s1")
        assert result["response"] == "Answer text"
        assert result["route"] == "finance_qa"
        assert result["sources"] == ["kb"]

    @patch("src.graph.get_app")
    def test_invoke_handles_missing_response_gracefully(self, mock_get_app):
        mock_app = MagicMock()
        mock_app.invoke.return_value = {}
        mock_get_app.return_value = mock_app

        result = invoke("test query", user_id="u1", session_id="s1")
        assert "couldn't generate" in result["response"].lower()
