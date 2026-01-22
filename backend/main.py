"""
AI Research Assistant - Production FastAPI Backend
Complete implementation with hybrid RAG and multi-agent system
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules
from rag.hybrid_retriever import HybridRetriever
from agents.research_agent import ResearchAgent
from evaluation.deepeval_suite import RAGEvaluator

# ===== DATA MODELS =====

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    retrieval_methods: List[str]
    scores: List[float]
    conversation_id: str
    timestamp: str

class DocumentUploadResponse(BaseModel):
    filename: str
    chunks: int
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_count: int

# ===== FASTAPI APP =====

app = FastAPI(
    title="AI Research Assistant API",
    description="Production-ready agentic RAG system with hybrid retrieval",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",  # Angular dev
        "http://localhost:3000",  # Alternative
        "https://*.vercel.app",   # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== GLOBAL INSTANCES =====

# Initialize components on startup
hybrid_retriever = None
research_agent = None
rag_evaluator = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global hybrid_retriever, research_agent, rag_evaluator
    
    logger.info("🚀 Starting AI Research Assistant API...")
    
    try:
        # Initialize hybrid retriever
        hybrid_retriever = HybridRetriever()
        logger.info("✅ Hybrid retriever initialized")
        
        # Initialize research agent
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("⚠️  ANTHROPIC_API_KEY not set - agent will not work")
        else:
            research_agent = ResearchAgent(
                retriever=hybrid_retriever,
                api_key=api_key
            )
            logger.info("✅ Research agent initialized")
        
        # Initialize evaluator
        rag_evaluator = RAGEvaluator()
        logger.info("✅ RAG evaluator initialized")
        
        logger.info("🎉 All systems ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {str(e)}")
        raise

# ===== ENDPOINTS =====

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "AI Research Assistant API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint with system status"""
    try:
        vector_count = hybrid_retriever.get_document_count() if hybrid_retriever else 0
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            vector_store_count=vector_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint using multi-agent system
    
    - Accepts a user message
    - Uses hybrid retrieval to find relevant documents
    - Generates response using Claude AI
    - Returns answer with sources and metadata
    """
    try:
        if not research_agent:
            raise HTTPException(
                status_code=500, 
                detail="Research agent not initialized. Check ANTHROPIC_API_KEY."
            )
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        logger.info(f"Processing query: {request.message[:50]}...")
        
        # Process query through research agent
        result = await research_agent.process_query(
            query=request.message,
            conversation_id=conversation_id
        )
        
        return ChatResponse(
            response=result['answer'],
            sources=result['sources'],
            retrieval_methods=result.get('retrieval_methods', []),
            scores=result.get('scores', []),
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process documents for RAG
    
    - Accepts PDF, TXT, DOCX files
    - Chunks documents intelligently
    - Adds to both semantic and BM25 indexes
    - Returns processing status
    """
    try:
        if not hybrid_retriever:
            raise HTTPException(status_code=500, detail="Retriever not initialized")
        
        # Validate file type
        allowed_types = ['.pdf', '.txt', '.docx', '.doc']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {allowed_types}"
            )
        
        logger.info(f"Processing file: {file.filename}")
        
        # Read file content
        content = await file.read()
        
        # Process and add to retriever
        chunks_count = await hybrid_retriever.ingest_document(
            content=content,
            filename=file.filename
        )
        
        logger.info(f"✅ Processed {file.filename}: {chunks_count} chunks")
        
        return DocumentUploadResponse(
            filename=file.filename,
            chunks=chunks_count,
            status="success",
            message=f"Successfully processed {chunks_count} chunks"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", tags=["Documents"])
async def list_documents():
    """
    List all uploaded documents and statistics
    """
    try:
        if not hybrid_retriever:
            raise HTTPException(status_code=500, detail="Retriever not initialized")
        
        stats = hybrid_retriever.get_statistics()
        
        return {
            "total_documents": stats['document_count'],
            "total_chunks": stats['chunk_count'],
            "vector_store_size": stats['vector_store_size'],
            "bm25_index_size": stats['bm25_index_size'],
            "status": "active"
        }
        
    except Exception as e:
        logger.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents", tags=["Documents"])
async def clear_documents():
    """
    Clear all documents from the system
    """
    try:
        if not hybrid_retriever:
            raise HTTPException(status_code=500, detail="Retriever not initialized")
        
        hybrid_retriever.clear_all()
        
        return {
            "status": "success",
            "message": "All documents cleared"
        }
        
    except Exception as e:
        logger.error(f"Clear documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", tags=["Evaluation"])
async def evaluate_system(test_queries: List[Dict[str, str]]):
    """
    Evaluate RAG system using DeepEval
    
    Expects: [{"question": "...", "ground_truth": "..."}]
    """
    try:
        if not rag_evaluator or not research_agent:
            raise HTTPException(status_code=500, detail="Evaluation system not ready")
        
        logger.info(f"Running evaluation on {len(test_queries)} queries...")
        
        results = await rag_evaluator.evaluate_queries(
            research_agent=research_agent,
            test_cases=test_queries
        )
        
        return {
            "status": "success",
            "test_cases": len(test_queries),
            "results": results['summary'],
            "average_scores": results['average_scores']
        }
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{conversation_id}", tags=["Chat"])
async def get_conversation_history(conversation_id: str):
    """
    Retrieve conversation history
    """
    try:
        if not research_agent:
            raise HTTPException(status_code=500, detail="Agent not initialized")
        
        history = research_agent.get_conversation_history(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "message_count": len(history),
            "messages": history
        }
        
    except Exception as e:
        logger.error(f"Get conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== RUN SERVER =====

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     AI Research Assistant - Production Backend           ║
    ║                                                          ║
    ║  Features:                                              ║
    ║  ✅ Hybrid Retrieval (Semantic + BM25)                   ║
    ║  ✅ Multi-Agent System (LangGraph)                       ║
    ║  ✅ DeepEval Evaluation                                  ║
    ║  ✅ Document Processing                                  ║
    ║  ✅ Conversation Memory                                  ║
    ╚══════════════════════════════════════════════════════════╝
    
    🚀 Starting server...
    📖 API Docs: http://localhost:8000/docs
    🔍 Health: http://localhost:8000/health
    
    Make sure you have set: ANTHROPIC_API_KEY
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )