from unittest.mock import patch, MagicMock
import pytest
import src.rag.vector_store as vs_module


@pytest.fixture(autouse=True)
def reset_singleton():
    """The module caches a singleton vectorstore; reset it between tests."""
    vs_module._vectorstore = None
    yield
    vs_module._vectorstore = None


class TestGetVectorstore:
    @patch("src.rag.vector_store.Chroma")
    def test_defaults_to_chroma(self, mock_chroma):
        vs_module.settings.VECTOR_BACKEND = "chroma"
        mock_instance = MagicMock()
        mock_chroma.return_value = mock_instance

        result = vs_module.get_vectorstore()
        assert result == mock_instance
        mock_chroma.assert_called_once()

    @patch("src.rag.vector_store.Chroma")
    def test_singleton_reused_on_second_call(self, mock_chroma):
        vs_module.settings.VECTOR_BACKEND = "chroma"
        mock_chroma.return_value = MagicMock()

        vs_module.get_vectorstore()
        vs_module.get_vectorstore()
        assert mock_chroma.call_count == 1  # only constructed once


class TestSimilaritySearch:
    @patch("src.rag.vector_store.get_retriever")
    def test_shapes_documents_correctly(self, mock_get_retriever):
        mock_doc = MagicMock()
        mock_doc.page_content = "Roth IRA content"
        mock_doc.metadata = {"source": "IRS", "category": "tax_education", "title": "Roth IRA"}

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_get_retriever.return_value = mock_retriever

        results = vs_module.similarity_search("What is a Roth IRA?", k=4, category="tax_education")
        assert len(results) == 1
        assert results[0]["content"] == "Roth IRA content"
        assert results[0]["source"] == "IRS"
        assert results[0]["category"] == "tax_education"

    @patch("src.rag.vector_store.get_retriever")
    def test_handles_missing_metadata_gracefully(self, mock_get_retriever):
        mock_doc = MagicMock()
        mock_doc.page_content = "Some content"
        mock_doc.metadata = {}

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_get_retriever.return_value = mock_retriever

        results = vs_module.similarity_search("query")
        assert results[0]["source"] == "unknown"
        assert results[0]["category"] == "general"


class TestGetRetriever:
    @patch("src.rag.vector_store.get_vectorstore")
    def test_applies_category_filter(self, mock_get_vs):
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs

        vs_module.get_retriever(k=3, category="tax_education")
        mock_vs.as_retriever.assert_called_once_with(
            search_kwargs={"k": 3, "filter": {"category": "tax_education"}}
        )

    @patch("src.rag.vector_store.get_vectorstore")
    def test_no_filter_without_category(self, mock_get_vs):
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs

        vs_module.get_retriever(k=4)
        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 4})
