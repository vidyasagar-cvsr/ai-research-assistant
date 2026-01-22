"""
Hybrid Retrieval System
Combines semantic search (ChromaDB) with BM25 keyword search
Uses Reciprocal Rank Fusion for optimal results
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import PyPDF2
from io import BytesIO

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    content: str
    score: float
    source: str
    method: str  # 'semantic', 'bm25', or 'hybrid'
    metadata: Dict[str, Any]


class HybridRetriever:
    """
    Production hybrid retrieval system
    
    Combines:
    - Semantic search using dense embeddings
    - BM25 keyword search
    - Reciprocal Rank Fusion for result combination
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        """
        Initialize hybrid retriever
        
        Args:
            embedding_model: HuggingFace model for embeddings
            persist_directory: Where to store ChromaDB
            semantic_weight: Weight for semantic search in RRF
            bm25_weight: Weight for BM25 search in RRF
        """
        logger.info("Initializing Hybrid Retriever...")
        
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Vector store for semantic search
        self.vector_store = Chroma(
            collection_name="research_docs",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        # BM25 index for keyword search
        self.bm25_index = None
        self.documents: List[Document] = []
        
        # Weights for hybrid fusion
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info("✅ Hybrid Retriever initialized")
    
    async def ingest_document(
        self,
        content: bytes,
        filename: str
    ) -> int:
        """
        Ingest document and add to both indexes
        
        Args:
            content: Document content as bytes
            filename: Original filename
            
        Returns:
            Number of chunks created
        """
        try:
            # Extract text based on file type
            text = self._extract_text(content, filename)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create metadata
            metadatas = [
                {
                    "source": filename,
                    "chunk": i,
                    "total_chunks": len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            # Add to vector store (semantic)
            self.vector_store.add_texts(texts=chunks, metadatas=metadatas)
            
            # Add to BM25 index (keyword)
            new_docs = [
                Document(page_content=chunk, metadata=meta)
                for chunk, meta in zip(chunks, metadatas)
            ]
            self.documents.extend(new_docs)
            
            # Rebuild BM25 index
            self._rebuild_bm25_index()
            
            logger.info(f"Ingested {filename}: {len(chunks)} chunks")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            raise
    
    def _extract_text(self, content: bytes, filename: str) -> str:
        """Extract text from different file types"""
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.pdf':
            return self._extract_from_pdf(content)
        elif file_ext in ['.txt', '.md']:
            return content.decode('utf-8', errors='ignore')
        elif file_ext in ['.docx', '.doc']:
            # TODO: Implement DOCX extraction
            return content.decode('utf-8', errors='ignore')
        else:
            # Try to decode as text
            return content.decode('utf-8', errors='ignore')
    
    def _extract_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            raise
    
    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from documents"""
        if not self.documents:
            self.bm25_index = None
            return
        
        tokenized_docs = [
            doc.page_content.lower().split()
            for doc in self.documents
        ]
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def _semantic_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Semantic search using vector embeddings"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            return []
    
    def _bm25_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Keyword search using BM25"""
        if not self.bm25_index or not self.documents:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion
        
        Standard algorithm used by Elastic, OpenSearch, etc.
        Score = semantic_weight/(k+rank) + bm25_weight/(k+rank)
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            k: RRF constant (typically 60)
            
        Returns:
            Fused and ranked results
        """
        doc_scores = {}
        
        # Process semantic results
        for rank, (doc, score) in enumerate(semantic_results, 1):
            doc_id = doc.page_content
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'score': 0,
                    'doc': doc,
                    'semantic_rank': rank,
                    'bm25_rank': None,
                    'semantic_score': score,
                    'bm25_score': None
                }
            doc_scores[doc_id]['score'] += self.semantic_weight / (k + rank)
        
        # Process BM25 results
        for rank, (doc, score) in enumerate(bm25_results, 1):
            doc_id = doc.page_content
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'score': 0,
                    'doc': doc,
                    'semantic_rank': None,
                    'bm25_rank': rank,
                    'semantic_score': None,
                    'bm25_score': score
                }
            else:
                doc_scores[doc_id]['bm25_rank'] = rank
                doc_scores[doc_id]['bm25_score'] = score
            
            doc_scores[doc_id]['score'] += self.bm25_weight / (k + rank)
        
        # Sort by combined score
        ranked = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Convert to RetrievalResult
        results = []
        for item in ranked:
            # Determine method
            if item['semantic_rank'] and item['bm25_rank']:
                method = 'hybrid'
            elif item['semantic_rank']:
                method = 'semantic'
            else:
                method = 'bm25'
            
            results.append(RetrievalResult(
                content=item['doc'].page_content,
                score=item['score'],
                source=item['doc'].metadata.get('source', 'unknown'),
                method=method,
                metadata=item['doc'].metadata
            ))
        
        return results
    
    def retrieve(
        self,
        query: str,
        k: int = 5
    ) -> List[RetrievalResult]:
        """
        Main retrieval method using hybrid search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            Top k results from hybrid search
        """
        logger.info(f"Retrieving for query: {query[:50]}...")
        
        # Get results from both methods
        semantic_results = self._semantic_search(query, k=k)
        bm25_results = self._bm25_search(query, k=k)
        
        logger.info(f"Semantic: {len(semantic_results)}, BM25: {len(bm25_results)}")
        
        # Combine with RRF
        hybrid_results = self._reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            k=60
        )
        
        # Return top k
        return hybrid_results[:k]
    
    def get_document_count(self) -> int:
        """Get total number of document chunks"""
        return len(self.documents)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        return {
            'document_count': len(set(doc.metadata.get('source') for doc in self.documents)),
            'chunk_count': len(self.documents),
            'vector_store_size': self.vector_store._collection.count() if self.vector_store else 0,
            'bm25_index_size': len(self.documents) if self.bm25_index else 0
        }
    
    def clear_all(self):
        """Clear all documents from both indexes"""
        self.documents = []
        self.bm25_index = None
        # Note: ChromaDB clear needs to be implemented
        logger.info("Cleared all documents")