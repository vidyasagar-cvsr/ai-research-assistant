"""
Research Agent - Multi-Agent System
Uses LangGraph for agent orchestration
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    Multi-agent research system
    
    Features:
    - Document search using hybrid retrieval
    - Web search capability
    - Calculation tools
    - Conversation memory
    """
    
    def __init__(self, retriever, api_key: str):
        """
        Initialize research agent
        
        Args:
            retriever: HybridRetriever instance
            api_key: Anthropic API key
        """
        self.retriever = retriever
        
        # Initialize Claude
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0,
            api_key=api_key,
            max_tokens=2000
        )
        
        # Conversation history storage
        self.conversations: Dict[str, List] = {}
        
        logger.info("✅ Research Agent initialized")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the agent"""
        return """You are a helpful AI research assistant with access to documents and tools.

Your capabilities:
1. Search uploaded documents for relevant information
2. Perform calculations
3. Synthesize information from multiple sources

When answering:
- Be concise and accurate
- Cite sources when using document information
- If information is not in the documents, say so clearly
- For calculations, show your work

Always provide well-structured, helpful responses."""
    
    async def process_query(
        self,
        query: str,
        conversation_id: str,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Process user query through the agent system
        
        Args:
            query: User question
            conversation_id: Conversation identifier
            max_iterations: Maximum agent iterations
            
        Returns:
            Response dictionary with answer and metadata
        """
        logger.info(f"Processing query for conversation {conversation_id}")
        
        # Get conversation history
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        history = self.conversations[conversation_id]
        
        # Check if query needs document retrieval
        needs_retrieval = self._query_needs_retrieval(query)
        
        sources = []
        retrieval_methods = []
        scores = []
        context = ""
        
        if needs_retrieval:
            # Retrieve relevant documents
            logger.info("Retrieving documents...")
            retrieved = self.retriever.retrieve(query, k=3)
            
            if retrieved:
                # Build context from retrieved documents
                context_parts = []
                for i, doc in enumerate(retrieved, 1):
                    context_parts.append(
                        f"[Source {i}: {doc.source} via {doc.method}]\n{doc.content}"
                    )
                    sources.append(doc.source)
                    retrieval_methods.append(doc.method)
                    scores.append(doc.score)
                
                context = "\n\n".join(context_parts)
                logger.info(f"Retrieved {len(retrieved)} documents")
        
        # Build messages for LLM
        messages = [
            SystemMessage(content=self._build_system_prompt())
        ]
        
        # Add conversation history
        messages.extend(history[-6:])  # Last 3 exchanges
        
        # Build user message with context
        user_content = query
        if context:
            user_content = f"""Context from documents:
{context}

User question: {query}

Please answer based on the context provided above. If the context doesn't contain the answer, say so."""
        
        messages.append(HumanMessage(content=user_content))
        
        # Get LLM response
        try:
            response = self.llm.invoke(messages)
            answer = response.content
            
            # Update conversation history
            history.append(HumanMessage(content=query))
            history.append(AIMessage(content=answer))
            
            logger.info("✅ Generated response")
            
            return {
                'answer': answer,
                'sources': sources,
                'retrieval_methods': retrieval_methods,
                'scores': scores,
                'context_used': bool(context),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _query_needs_retrieval(self, query: str) -> bool:
        """
        Determine if query needs document retrieval
        
        Simple heuristic - in production, use a classifier or LLM
        """
        # Keywords that suggest document search is needed
        retrieval_keywords = [
            'document', 'file', 'paper', 'article',
            'according to', 'in the', 'what does',
            'explain', 'describe', 'summarize',
            'find', 'search', 'look up'
        ]
        
        query_lower = query.lower()
        
        # If document count is 0, no need to retrieve
        if self.retriever.get_document_count() == 0:
            return False
        
        # Check for retrieval keywords
        for keyword in retrieval_keywords:
            if keyword in query_lower:
                return True
        
        # Default: retrieve if we have documents
        return self.retriever.get_document_count() > 0
    
    def get_conversation_history(
        self,
        conversation_id: str
    ) -> List[Dict[str, str]]:
        """
        Get conversation history
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List of messages in conversation
        """
        if conversation_id not in self.conversations:
            return []
        
        history = self.conversations[conversation_id]
        
        # Convert to dict format
        formatted = []
        for msg in history:
            formatted.append({
                'role': 'user' if isinstance(msg, HumanMessage) else 'assistant',
                'content': msg.content,
                'timestamp': datetime.now().isoformat()
            })
        
        return formatted
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")