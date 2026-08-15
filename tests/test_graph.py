from unittest.mock import patch, MagicMock
import tempfile
import os
from langgraph.checkpoint.memory import MemorySaver
from src.graph import build_graph, invoke, _build_default_checkpointer


class TestCheckpointerSelection:
    def test_defaults_to_memory_saver(self):
        with patch("src.graph.settings") as mock_settings:
            mock_settings.CHECKPOINTER_BACKEND = "memory"
            checkpointer = _build_default_checkpointer()
            assert isinstance(checkpointer, MemorySaver)

    def test_selects_sqlite_saver_when_configured(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test_checkpoints.sqlite")
            with patch("src.graph.settings") as mock_settings:
                mock_settings.CHECKPOINTER_BACKEND = "sqlite"
                mock_settings.SQLITE_CHECKPOINT_PATH = db_path
                checkpointer = _build_default_checkpointer()
                assert isinstance(checkpointer, SqliteSaver)
                assert os.path.exists(db_path)

    def test_sqlite_backend_persists_across_separate_graph_instances(self):
        """
        The core guarantee this feature exists for: state written by one
        compiled graph instance (simulating one app process) must be
        readable by a second, independently-built graph instance pointed
        at the same database file (simulating a restart).
        """
        from langgraph.checkpoint.sqlite import SqliteSaver
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test_checkpoints.sqlite")
            config = {"configurable": {"thread_id": "persist-test"}}

            import sqlite3
            conn1 = sqlite3.connect(db_path, check_same_thread=False)
            saver1 = SqliteSaver(conn1)
            saver1.setup()
            graph1 = build_graph(checkpointer=saver1)
            with patch("src.router._router_llm") as mock_router, \
                 patch("src.agents.finance_qa_agent.get_llm") as mock_llm, \
                 patch("src.agents.finance_qa_agent.retrieve_context") as mock_ctx:
                mock_router.invoke.return_value = MagicMock(route="finance_qa", confidence=0.9)
                mock_ctx.return_value = []
                mock_llm.return_value = MagicMock(
                    invoke=MagicMock(return_value=MagicMock(content="persisted answer"))
                )
                graph1.invoke(
                    {"messages": [], "user_id": "u1", "session_id": "persist-test",
                     "query": "test query", "portfolio": []},
                    config=config,
                )
            conn1.close()

            # Fresh connection + fresh compiled graph, simulating a restart
            conn2 = sqlite3.connect(db_path, check_same_thread=False)
            saver2 = SqliteSaver(conn2)
            graph2 = build_graph(checkpointer=saver2)
            recovered = graph2.get_state(config)
            assert recovered.values.get("final_response") == "persisted answer"
            conn2.close()


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
