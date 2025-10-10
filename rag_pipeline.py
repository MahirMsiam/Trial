from typing import Dict, List, Optional, Iterator
from rag_retriever import HybridRetriever
from llm_client import get_llm_client, LLMClient, validate_response
from conversation_manager import ConversationManager
import prompt_templates
from config import CRIME_KEYWORDS
import logging_config  # noqa: F401
import logging

# Get logger
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline orchestrating retrieval, generation, and conversation management."""
    
    def __init__(self):
        """Initialize the RAG pipeline with all required components."""
        logger.info("Initializing RAG Pipeline...")
        
        # Initialize components
        self.retriever = HybridRetriever()
        self.llm_client = get_llm_client()
        self.conversation_manager = ConversationManager()
        
        logger.info("✅ RAG Pipeline initialized successfully")
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of user query.
        
        Args:
            query: User query string
            
        Returns:
            Query type: 'crime_search', 'comparison', 'summarization', 'factual', 'general'
        """
        query_lower = query.lower()
        
        # Check for crime keywords
        for crime_type, keywords in CRIME_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    if any(word in query_lower for word in ['show', 'list', 'find', 'all', 'cases']):
                        return 'crime_search'
        
        # Check for comparison queries
        if any(word in query_lower for word in ['compare', 'difference', 'similar', 'contrast']):
            return 'comparison'
        
        # Check for summarization queries
        if any(word in query_lower for word in ['summarize', 'summary', 'brief', 'overview']):
            return 'summarization'
        
        # Default to factual
        return 'factual'
    
    def _validate_retrieval_quality(self, contexts: List[Dict], query: str) -> bool:
        """
        Validate that retrieved contexts are of sufficient quality.
        
        Args:
            contexts: Retrieved context chunks
            query: Original query
            
        Returns:
            True if quality is sufficient
        """
        if not contexts:
            return False
        
        # Check if at least one context has good similarity
        has_good_match = any(
            ctx.get('similarity', 0) > 0.7 or ctx.get('hybrid_score', 0) > 0.5
            for ctx in contexts
        )
        
        return has_good_match
    
    def _handle_no_results(self, query: str) -> Dict:
        """
        Generate helpful response when no relevant contexts found.
        
        Args:
            query: Original query
            
        Returns:
            Response dict with fallback message
        """
        return {
            "response": prompt_templates.FALLBACK_RESPONSE,
            "sources": [],
            "query_type": "no_results",
            "session_id": None
        }
    
    def process_query(self, query: str, session_id: str = None, filters: Dict = None) -> Dict:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query: User's question
            session_id: Optional conversation session ID
            filters: Optional search filters
            
        Returns:
            Dict with response, sources, and metadata
        """
        logger.info(f"Processing query: '{query}'")
        
        try:
            # Detect query type
            query_type = self._detect_query_type(query)
            logger.info(f"Query type detected: {query_type}")
            
            # Handle crime search specially
            if query_type == 'crime_search':
                return self.search_crime_cases(query)
            
            # Retrieve relevant contexts
            contexts = self.retriever.hybrid_retrieve(query, top_k=5, filters=filters)
            
            # Validate retrieval quality
            if not self._validate_retrieval_quality(contexts, query):
                logger.warning("Retrieved contexts do not meet quality threshold")
                return self._handle_no_results(query)
            
            # Get conversation history if session provided
            conversation_history = []
            if session_id:
                try:
                    conversation_history = self.conversation_manager.get_context_for_prompt(session_id)
                except ValueError:
                    logger.warning(f"Session {session_id} not found, creating new session")
                    session_id = self.conversation_manager.create_session()
            else:
                session_id = self.conversation_manager.create_session()
            
            # Build prompt based on query type
            if query_type == 'comparison':
                user_prompt = prompt_templates.build_comparison_prompt(query, contexts)
            elif query_type == 'summarization' and len(contexts) == 1:
                full_case = self.retriever.get_full_judgment(contexts[0]['case_id'])
                user_prompt = prompt_templates.build_summarization_prompt(full_case)
            else:
                user_prompt = prompt_templates.build_rag_prompt(query, contexts, conversation_history)
            
            # Get system prompt
            system_prompt = prompt_templates.get_system_prompt()
            
            # Generate response
            logger.info("Generating LLM response...")
            response = self.llm_client.generate(user_prompt, system_prompt)
            
            # Validate response
            if not validate_response(response):
                logger.warning("LLM response failed validation")
                response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            # Format response with citations
            formatted_response = prompt_templates.format_response_with_citations(response, contexts)
            
            # Update conversation history with formatted response
            self.conversation_manager.add_message(session_id, "user", query)
            self.conversation_manager.add_message(session_id, "assistant", formatted_response)
            
            return {
                "response": formatted_response,
                "sources": contexts,
                "query_type": query_type,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "query_type": "error",
                "session_id": session_id
            }
    
    def process_query_stream(self, query: str, session_id: str = None, filters: Dict = None) -> Iterator[Dict]:
        """
        Process query with streaming response.
        
        Args:
            query: User's question
            session_id: Optional conversation session ID
            filters: Optional search filters
            
        Yields:
            Chunks of response data
        """
        logger.info(f"Processing streaming query: '{query}'")
        
        try:
            # Detect query type
            query_type = self._detect_query_type(query)
            
            # Retrieve contexts
            contexts = self.retriever.hybrid_retrieve(query, top_k=5, filters=filters)
            
            # Yield sources first
            yield {"type": "sources", "content": contexts}
            
            if not self._validate_retrieval_quality(contexts, query):
                yield {"type": "complete", "content": self._handle_no_results(query)}
                return
            
            # Get conversation history
            conversation_history = []
            if session_id:
                try:
                    conversation_history = self.conversation_manager.get_context_for_prompt(session_id)
                except ValueError:
                    session_id = self.conversation_manager.create_session()
            else:
                session_id = self.conversation_manager.create_session()
            
            # Build prompt
            user_prompt = prompt_templates.build_rag_prompt(query, contexts, conversation_history)
            system_prompt = prompt_templates.get_system_prompt()
            
            # Stream response
            full_response = ""
            for token in self.llm_client.generate_stream(user_prompt, system_prompt):
                full_response += token
                yield {"type": "token", "content": token}
            
            # Apply citation formatting
            formatted_response = prompt_templates.format_response_with_citations(full_response, contexts)
            
            # Update conversation history
            self.conversation_manager.add_message(session_id, "user", query)
            self.conversation_manager.add_message(session_id, "assistant", formatted_response)
            
            # Yield completion with formatted response
            yield {
                "type": "complete",
                "content": {
                    "response": formatted_response,
                    "query_type": query_type,
                    "session_id": session_id
                }
            }
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield {"type": "error", "content": str(e)}
    
    def search_crime_cases(self, query: str, limit: int = 20) -> Dict:
        """
        Search for cases by crime category.
        
        Args:
            query: Query containing crime keywords
            limit: Maximum number of cases to return
            
        Returns:
            Dict with crime cases and summary
        """
        logger.info(f"Searching crime cases for: '{query}'")
        
        # Detect crime type
        detected_crime = None
        query_lower = query.lower()
        for crime_type, keywords in CRIME_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    detected_crime = crime_type
                    break
            if detected_crime:
                break
        
        if not detected_crime:
            return self._handle_no_results(query)
        
        # Retrieve cases
        cases = self.retriever.retrieve_by_crime_category(query)[:limit]
        
        if not cases:
            return {
                "response": f"No cases found for crime type: {detected_crime}",
                "crime_type": detected_crime,
                "count": 0,
                "cases": [],
                "summary": None
            }
        
        # Generate summary using LLM
        crime_prompt = prompt_templates.build_crime_category_prompt(query, detected_crime, cases)
        system_prompt = prompt_templates.get_system_prompt()
        
        summary = self.llm_client.generate(crime_prompt, system_prompt)
        
        return {
            "response": summary,
            "crime_type": detected_crime,
            "count": len(cases),
            "cases": cases,
            "summary": summary
        }
    
    def compare_cases(self, case_ids: List[int]) -> Dict:
        """
        Compare multiple specific cases.
        
        Args:
            case_ids: List of case IDs to compare
            
        Returns:
            Dict with comparison analysis
        """
        logger.info(f"Comparing cases: {case_ids}")
        
        # Retrieve full case data
        cases = [self.retriever.get_full_judgment(cid) for cid in case_ids]
        cases = [c for c in cases if c]  # Filter out empty results
        
        if len(cases) < 2:
            return {
                "response": "Need at least 2 valid cases to compare",
                "cases": cases
            }
        
        # Build comparison prompt
        comparison_prompt = prompt_templates.build_comparison_prompt("Compare these cases", cases)
        system_prompt = prompt_templates.get_system_prompt()
        
        # Generate comparison
        comparison = self.llm_client.generate(comparison_prompt, system_prompt)
        
        return {
            "comparison": comparison,
            "cases": cases
        }
    
    def summarize_case(self, case_id: int) -> Dict:
        """
        Generate detailed summary of a specific judgment.
        
        Args:
            case_id: Database ID of the case
            
        Returns:
            Dict with summary and case data
        """
        logger.info(f"Summarizing case: {case_id}")
        
        # Retrieve full case data
        case_data = self.retriever.get_full_judgment(case_id)
        
        if not case_data:
            return {
                "response": f"Case with ID {case_id} not found",
                "case_data": None
            }
        
        # Build summarization prompt
        summary_prompt = prompt_templates.build_summarization_prompt(case_data)
        system_prompt = prompt_templates.get_system_prompt()
        
        # Generate summary
        summary = self.llm_client.generate(summary_prompt, system_prompt)
        
        return {
            "summary": summary,
            "case_data": case_data
        }


if __name__ == '__main__':
    # Test the RAG pipeline
    print("Testing RAG Pipeline...")
    
    try:
        pipeline = RAGPipeline()
        
        test_query = "What are the legal principles for murder cases?"
        print(f"\nTest Query: '{test_query}'")
        
        result = pipeline.process_query(test_query)
        
        print(f"\nQuery Type: {result['query_type']}")
        print(f"Session ID: {result['session_id']}")
        print(f"Sources: {len(result['sources'])}")
        print(f"\nResponse:\n{result['response'][:500]}...")
        
        print("\n✅ RAG Pipeline test complete")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
