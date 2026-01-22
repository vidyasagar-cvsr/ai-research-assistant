"""
Unit tests for Hybrid Retriever
"""

import pytest
from rag.hybrid_retriever import HybridRetriever, RetrievalResult


@pytest.fixture
def retriever():
    """Create retriever instance for testing"""
    return HybridRetriever(
        persist_directory="./test_chroma_db"
    )


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        "LangChain is a framework for developing applications powered by language models.",
        "RAG combines retrieval and generation for better LLM responses.",
        "Vector databases store embeddings for semantic search.",
        "BM25 is a keyword-based search algorithm.",
        "Hybrid search combines semantic and keyword methods."
    ]


class TestHybridRetriever:
    """Test suite for HybridRetriever"""
    
    def test_initialization(self, retriever):
        """Test retriever initializes correctly"""
        assert retriever is not None
        assert retriever.embeddings is not None
        assert retriever.vector_store is not None
    
    def test_document_ingestion(self, retriever, sample_documents):
        """Test document ingestion"""
        # Add documents
        for doc in sample_documents:
            content = doc.encode('utf-8')
            chunks = retriever.ingest_document(content, "test.txt")
            assert chunks > 0
        
        # Check document count
        assert retriever.get_document_count() > 0
    
    def test_semantic_search(self, retriever, sample_documents):
        """Test semantic search"""
        # Add documents first
        for doc in sample_documents:
            content = doc.encode('utf-8')
            retriever.ingest_document(content, "test.txt")
        
        # Search
        results = retriever._semantic_search("What is RAG?", k=3)
        
        assert len(results) > 0
        assert all(isinstance(r[0].page_content, str) for r in results)
    
    def test_bm25_search(self, retriever, sample_documents):
        """Test BM25 search"""
        # Add documents
        for doc in sample_documents:
            content = doc.encode('utf-8')
            retriever.ingest_document(content, "test.txt")
        
        # Search
        results = retriever._bm25_search("BM25 keyword", k=3)
        
        assert len(results) > 0
    
    def test_hybrid_retrieval(self, retriever, sample_documents):
        """Test hybrid retrieval with RRF"""
        # Add documents
        for doc in sample_documents:
            content = doc.encode('utf-8')
            retriever.ingest_document(content, "test.txt")
        
        # Retrieve
        results = retriever.retrieve("hybrid search methods", k=3)
        
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.method in ['semantic', 'bm25', 'hybrid'] for r in results)
        assert all(r.score > 0 for r in results)
    
    def test_statistics(self, retriever, sample_documents):
        """Test statistics retrieval"""
        # Add documents
        for doc in sample_documents:
            content = doc.encode('utf-8')
            retriever.ingest_document(content, "test.txt")
        
        stats = retriever.get_statistics()
        
        assert 'document_count' in stats
        assert 'chunk_count' in stats
        assert stats['chunk_count'] > 0


@pytest.mark.asyncio
async def test_async_ingestion(retriever):
    """Test async document ingestion"""
    content = b"Test document content for async ingestion"
    chunks = await retriever.ingest_document(content, "async_test.txt")
    
    assert chunks > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])