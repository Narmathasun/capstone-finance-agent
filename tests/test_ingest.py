import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import src.rag.ingest as ingest_module
from src.rag.ingest import _parse_frontmatter, load_documents, ingest


class TestParseFrontmatter:
    def test_parses_full_frontmatter(self):
        text = (
            "---\n"
            "title: Test Article\n"
            "source: Test Source\n"
            "category: finance_qa\n"
            "---\n"
            "Body content here."
        )
        meta, body = _parse_frontmatter(text, fallback_category="x", fallback_title="y")
        assert meta["title"] == "Test Article"
        assert meta["source"] == "Test Source"
        assert meta["category"] == "finance_qa"
        assert body == "Body content here."

    def test_missing_frontmatter_uses_fallback(self):
        text = "Just plain body content, no frontmatter."
        meta, body = _parse_frontmatter(text, fallback_category="tax_education", fallback_title="untitled")
        assert meta["category"] == "tax_education"
        assert meta["title"] == "untitled"
        assert body == text.strip()


class TestLoadDocuments:
    def test_loads_articles_from_category_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb_dir = Path(tmp) / "knowledge_base"
            cat_dir = kb_dir / "finance_qa"
            cat_dir.mkdir(parents=True)
            (cat_dir / "article1.md").write_text(
                "---\ntitle: Test\nsource: Internal\ncategory: finance_qa\n---\nContent body."
            )

            with patch.object(ingest_module, "KB_DIR", kb_dir):
                docs = load_documents()
            assert len(docs) == 1
            assert docs[0].metadata["category"] == "finance_qa"
            assert "Content body." in docs[0].page_content

    def test_empty_dir_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb_dir = Path(tmp) / "does_not_exist"
            with patch.object(ingest_module, "KB_DIR", kb_dir):
                docs = load_documents()
            assert docs == []


class TestIngest:
    @patch("src.rag.ingest.get_vectorstore")
    @patch("src.rag.ingest.load_documents")
    def test_ingest_adds_chunks_to_vectorstore(self, mock_load_docs, mock_get_vs):
        from langchain_core.documents import Document
        mock_load_docs.return_value = [
            Document(page_content="A" * 2000, metadata={"category": "finance_qa", "source": "test"})
        ]
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs

        count = ingest(chunk_size=800, chunk_overlap=100)
        assert count > 0
        mock_vs.add_documents.assert_called_once()

    @patch("src.rag.ingest.load_documents")
    def test_ingest_returns_zero_when_no_docs(self, mock_load_docs):
        mock_load_docs.return_value = []
        count = ingest()
        assert count == 0
