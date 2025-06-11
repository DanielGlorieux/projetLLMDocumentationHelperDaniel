# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from dotenv import load_dotenv
# import os
# import logging
# from typing import Any, Dict, List
#
# # Load environment variables
# load_dotenv()
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Import your existing core functions
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain import hub
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.history_aware_retriever import create_history_aware_retriever
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# import pinecone
#
# # Import constants (assuming you have a consts.py file)
# try:
#     from consts import INDEX_NAME
# except ImportError:
#     INDEX_NAME = "test"  # Default fallback
#
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes
#
# # Global variables for caching
# llm_instance = None
# embeddings_instance = None
# vector_store_instance = None
# pinecone_client = None
#
#
# def get_pinecone_client():
#     """Get Pinecone client instance"""
#     global pinecone_client
#     if pinecone_client is None:
#         api_key = os.getenv("PINECONE_API_KEY")
#         if not api_key:
#             raise ValueError("PINECONE_API_KEY not found in environment variables")
#
#         pinecone_client = pinecone.Pinecone(api_key=api_key)
#     return pinecone_client
#
#
# def get_gemini_llm():
#     """Get Gemini LLM instance with caching"""
#     global llm_instance
#     if llm_instance is None:
#         api_key = os.getenv("GOOGLE_API_KEY")
#         if not api_key:
#             raise ValueError("GOOGLE_API_KEY not found in environment variables")
#
#         llm_instance = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             google_api_key=api_key,
#             temperature=0.1,
#             convert_system_message_to_human=True
#         )
#     return llm_instance
#
#
# def get_gemini_embeddings():
#     """Get Gemini embeddings instance with caching"""
#     global embeddings_instance
#     if embeddings_instance is None:
#         api_key = os.getenv("GOOGLE_API_KEY")
#         if not api_key:
#             raise ValueError("GOOGLE_API_KEY not found in environment variables")
#
#         embeddings_instance = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=api_key
#         )
#     return embeddings_instance
#
#
# def get_vector_store():
#     """Get vector store instance with caching and validation"""
#     global vector_store_instance
#     if vector_store_instance is None:
#         try:
#             embeddings = get_gemini_embeddings()
#
#             # Initialize Pinecone client
#             pc = get_pinecone_client()
#
#             # Check if index exists
#             existing_indexes = pc.list_indexes()
#             index_names = [idx.name for idx in existing_indexes]
#
#             if INDEX_NAME not in index_names:
#                 logger.error(f"Index '{INDEX_NAME}' not found. Available indexes: {index_names}")
#                 raise ValueError(f"Index '{INDEX_NAME}' does not exist in Pinecone")
#
#             # Connect to the index
#             index = pc.Index(INDEX_NAME)
#
#             # Check index stats
#             stats = index.describe_index_stats()
#             logger.info(f"Index stats: {stats}")
#
#             if stats.total_vector_count == 0:
#                 logger.warning("Index exists but contains no vectors!")
#
#             vector_store_instance = PineconeVectorStore(
#                 index_name=INDEX_NAME,
#                 embedding=embeddings,
#                 pinecone_api_key=os.getenv("PINECONE_API_KEY")
#             )
#
#         except Exception as e:
#             logger.error(f"Failed to initialize vector store: {str(e)}")
#             raise
#
#     return vector_store_instance
#
#
# def debug_retrieval(query: str, k: int = 4):
#     """Debug function to check document retrieval with enhanced logging"""
#     try:
#         docsearch = get_vector_store()
#
#         # Try different search parameters
#         logger.info(f"Attempting similarity search for query: '{query}' with k={k}")
#
#         # Method 1: Standard similarity search
#         docs = docsearch.similarity_search(query, k=k)
#         logger.info(f"Standard search retrieved {len(docs)} documents")
#
#         if not docs:
#             # Method 2: Try with different parameters
#             logger.info("No documents found, trying similarity search with score threshold")
#             docs_with_scores = docsearch.similarity_search_with_score(query, k=k)
#             docs = [doc for doc, score in docs_with_scores]
#             logger.info(f"Search with scores retrieved {len(docs)} documents")
#
#             if docs_with_scores:
#                 logger.info(
#                     f"Score range: {min(score for _, score in docs_with_scores)} to {max(score for _, score in docs_with_scores)}")
#
#         if not docs:
#             # Method 3: Try broader search terms
#             logger.info("Trying broader search with individual words")
#             words = query.lower().split()
#             for word in words:
#                 if len(word) > 3:  # Skip very short words
#                     word_docs = docsearch.similarity_search(word, k=2)
#                     docs.extend(word_docs)
#                     logger.info(f"Word '{word}' found {len(word_docs)} documents")
#                     if len(docs) >= k:
#                         docs = docs[:k]  # Limit to requested number
#                         break
#
#         # Log document details
#         for i, doc in enumerate(docs):
#             logger.info(f"Doc {i + 1}: {len(doc.page_content)} chars, metadata: {doc.metadata}")
#
#         return docs
#
#     except Exception as e:
#         logger.error(f"Error in debug_retrieval: {str(e)}")
#         return []
#
#
# def run_gemini_llm_with_debug(query: str, chat_history: List[Dict[str, Any]] = []):
#     """Main RAG function with enhanced debug information"""
#     try:
#         # Debug: Check what documents are retrieved
#         debug_docs = debug_retrieval(query, k=6)
#
#         if not debug_docs:
#             logger.warning(f"No documents retrieved for query: '{query}'. Falling back to LLM-only response.")
#             return run_gemini_fallback_only(query, chat_history)
#
#         # Get instances
#         docsearch = get_vector_store()
#         chat = get_gemini_llm()
#
#         # Get prompts from LangChain hub
#         try:
#             rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
#             retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
#         except Exception as e:
#             logger.error(f"Failed to pull prompts from hub: {e}")
#             # Use fallback prompts
#             return run_gemini_llm_fallback(query, chat_history)
#
#         # Create chains
#         stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
#
#         history_aware_retriever = create_history_aware_retriever(
#             llm=chat,
#             retriever=docsearch.as_retriever(search_kwargs={"k": 6}),
#             prompt=rephrase_prompt
#         )
#
#         qa = create_retrieval_chain(
#             retriever=history_aware_retriever,
#             combine_docs_chain=stuff_documents_chain
#         )
#
#         result = qa.invoke(input={"input": query, "chat_history": chat_history})
#
#         # Validate result has context
#         if not result.get("context") or len(result.get("context", [])) == 0:
#             logger.warning("Chain returned empty context, using fallback")
#             return run_gemini_llm_fallback(query, chat_history)
#
#         # Add debug information
#         result["debug_info"] = {
#             "retrieved_docs_count": len(debug_docs),
#             "has_context": len(result.get("context", [])) > 0,
#             "context_docs_count": len(result.get("context", [])),
#             "retrieval_method": "langchain_chain"
#         }
#
#         return result
#
#     except Exception as e:
#         logger.error(f"Error in run_gemini_llm_with_debug: {str(e)}")
#         return run_gemini_llm_fallback(query, chat_history)
#
#
# def run_gemini_fallback_only(query: str, chat_history: List[Dict[str, Any]] = []):
#     """LLM-only response when no documents are found"""
#     try:
#         chat = get_gemini_llm()
#
#         prompt_text = f"""Question: {query}
#
# Note: I was unable to find specific documents in my knowledge base related to your question.
# I'll provide a general answer based on my training knowledge, but please note that this response
# is not based on the specific documents in my knowledge base.
#
# Please provide a helpful and accurate response to the question above."""
#
#         response = chat.invoke(prompt_text)
#
#         return {
#             "answer": response.content if hasattr(response, 'content') else str(response),
#             "context": [],
#             "source_documents": [],
#             "debug_info": {
#                 "retrieval_strategy": "llm_only",
#                 "docs_found": 0,
#                 "used_context": False,
#                 "note": "No documents found in vector store"
#             }
#         }
#     except Exception as e:
#         logger.error(f"Error in run_gemini_fallback_only: {str(e)}")
#         raise
#
#
# def run_gemini_llm_fallback(query: str, chat_history: List[Dict[str, Any]] = []):
#     """Fallback RAG function with multiple search strategies"""
#     try:
#         docsearch = get_vector_store()
#         chat = get_gemini_llm()
#
#         docs = []
#
#         # Strategy 1: Direct similarity search with different k values
#         for k_val in [6, 10, 15]:
#             try:
#                 docs = docsearch.similarity_search(query, k=k_val)
#                 logger.info(f"Direct search with k={k_val} found {len(docs)} documents")
#                 if docs:
#                     break
#             except Exception as e:
#                 logger.error(f"Direct search with k={k_val} failed: {e}")
#
#         # Strategy 2: Search with score threshold
#         if not docs:
#             try:
#                 docs_with_scores = docsearch.similarity_search_with_score(query, k=10)
#                 # Filter by score threshold (adjust as needed)
#                 docs = [doc for doc, score in docs_with_scores if score < 1.5]
#                 logger.info(f"Score-based search found {len(docs)} documents")
#             except Exception as e:
#                 logger.error(f"Score-based search failed: {e}")
#
#         # Strategy 3: Broader search if no docs found
#         if not docs:
#             key_terms = query.lower().split()
#             for term in key_terms:
#                 if len(term) > 3:
#                     try:
#                         term_docs = docsearch.similarity_search(term, k=3)
#                         docs.extend(term_docs)
#                         logger.info(f"Term search for '{term}' found {len(term_docs)} documents")
#                         if len(docs) >= 6:
#                             docs = docs[:6]  # Limit total docs
#                             break
#                     except Exception as e:
#                         logger.error(f"Term search for '{term}' failed: {e}")
#
#         # Format context
#         context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
#
#         # Create prompt based on context availability
#         if context:
#             prompt_text = f"""Context information:
# {context}
#
# Question: {query}
#
# Please provide a comprehensive answer based on the context above. If the context doesn't fully address the question, please indicate what aspects are covered and what might be missing."""
#         else:
#             logger.warning("No context found, using LLM-only response")
#             return run_gemini_fallback_only(query, chat_history)
#
#         # Get response
#         response = chat.invoke(prompt_text)
#
#         return {
#             "answer": response.content if hasattr(response, 'content') else str(response),
#             "context": docs,
#             "source_documents": docs,
#             "debug_info": {
#                 "retrieval_strategy": "fallback_with_context",
#                 "docs_found": len(docs),
#                 "used_context": bool(context)
#             }
#         }
#
#     except Exception as e:
#         logger.error(f"Error in run_gemini_llm_fallback: {str(e)}")
#         return run_gemini_fallback_only(query, chat_history)
#
#
# # API ENDPOINTS
#
# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         "status": "healthy",
#         "message": "RAG API is running",
#         "timestamp": str(os.times())
#     })
#
#
# @app.route('/api/chat', methods=['POST'])
# def chat():
#     """Main chat endpoint for RAG queries"""
#     try:
#         data = request.get_json()
#
#         if not data or 'query' not in data:
#             return jsonify({
#                 "error": "Missing 'query' in request body"
#             }), 400
#
#         query = data['query']
#         chat_history = data.get('chat_history', [])
#
#         logger.info(f"Processing query: '{query[:100]}...' with {len(chat_history)} history items")
#
#         # Try main RAG function first, fallback if needed
#         try:
#             result = run_gemini_llm_with_debug(query, chat_history)
#         except Exception as e:
#             logger.warning(f"Primary RAG failed: {e}. Using fallback.")
#             result = run_gemini_llm_fallback(query, chat_history)
#
#         # Ensure we have an answer
#         if not result.get("answer"):
#             result = run_gemini_fallback_only(query, chat_history)
#
#         # Format response
#         response = {
#             "answer": result.get("answer", "I apologize, but I was unable to generate a response."),
#             "sources": [
#                 {
#                     "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
#                     "metadata": doc.metadata
#                 } for doc in result.get("context", [])
#             ],
#             "debug_info": result.get("debug_info", {}),
#             "success": True
#         }
#
#         logger.info(f"Response generated with {len(response['sources'])} sources")
#         return jsonify(response)
#
#     except Exception as e:
#         logger.error(f"Error in chat endpoint: {str(e)}")
#         return jsonify({
#             "error": str(e),
#             "success": False
#         }), 500
#
#
# @app.route('/api/search', methods=['POST'])
# def search_documents():
#     """Search documents in vector store"""
#     try:
#         data = request.get_json()
#
#         if not data or 'query' not in data:
#             return jsonify({
#                 "error": "Missing 'query' in request body"
#             }), 400
#
#         query = data['query']
#         k = data.get('k', 4)  # Number of documents to retrieve
#
#         docs = debug_retrieval(query, k)
#
#         response = {
#             "query": query,
#             "documents": [
#                 {
#                     "content": doc.page_content,
#                     "metadata": doc.metadata,
#                     "content_length": len(doc.page_content)
#                 } for doc in docs
#             ],
#             "total_documents": len(docs),
#             "success": True
#         }
#
#         return jsonify(response)
#
#     except Exception as e:
#         logger.error(f"Error in search endpoint: {str(e)}")
#         return jsonify({
#             "error": str(e),
#             "success": False
#         }), 500
#
#
# @app.route('/api/status', methods=['GET'])
# def system_status():
#     """Get system status and configuration with enhanced checks"""
#     try:
#         status_details = {}
#
#         # Test Gemini connection
#         try:
#             chat = get_gemini_llm()
#             test_response = chat.invoke("Hello")
#             gemini_status = True
#             status_details["gemini_test"] = "Success"
#         except Exception as e:
#             gemini_status = False
#             status_details["gemini_error"] = str(e)
#             logger.error(f"Gemini test failed: {e}")
#
#         # Test Vector Store connection
#         try:
#             docsearch = get_vector_store()
#
#             # Get Pinecone stats
#             pc = get_pinecone_client()
#             index = pc.Index(INDEX_NAME)
#             stats = index.describe_index_stats()
#
#             test_docs = docsearch.similarity_search("test", k=1)
#             vector_store_status = {
#                 "connected": True,
#                 "has_documents": len(test_docs) > 0,
#                 "sample_count": len(test_docs),
#                 "total_vectors": stats.total_vector_count,
#                 "index_stats": stats
#             }
#             status_details["vector_store_test"] = "Success"
#         except Exception as e:
#             vector_store_status = {
#                 "connected": False,
#                 "error": str(e)
#             }
#             status_details["vector_store_error"] = str(e)
#
#         response = {
#             "system_status": "operationel" if gemini_status and vector_store_status.get("connected",
#                                                                                         False) else "degraded",
#             "components": {
#                 "gemini_llm": {
#                     "status": "up" if gemini_status else "down",
#                     "model": "gemini-1.5-flash"
#                 },
#                 "vector_store": {
#                     "status": "up" if vector_store_status.get("connected", False) else "down",
#                     "index_name": INDEX_NAME,
#                     "details": vector_store_status
#                 }
#             },
#             "configuration": {
#                 "index_name": INDEX_NAME,
#                 "embedding_model": "models/embedding-001",
#                 "llm_model": "gemini-1.5-flash"
#             },
#             "status_details": status_details
#         }
#
#         return jsonify(response)
#
#     except Exception as e:
#         logger.error(f"Error in status endpoint: {str(e)}")
#         return jsonify({
#             "error": str(e),
#             "system_status": "error"
#         }), 500
#
#
# @app.route('/api/debug/index', methods=['GET'])
# def debug_index():
#     """Debug endpoint to check index contents"""
#     try:
#         pc = get_pinecone_client()
#         index = pc.Index(INDEX_NAME)
#
#         # Get index statistics
#         stats = index.describe_index_stats()
#
#         # Try to fetch a few vectors to verify content
#         # Note: This requires knowing some vector IDs, so we'll just return stats
#
#         return jsonify({
#             "index_name": INDEX_NAME,
#             "stats": stats,
#             "total_vectors": stats.total_vector_count,
#             "success": True
#         })
#
#     except Exception as e:
#         logger.error(f"Error in debug index endpoint: {str(e)}")
#         return jsonify({
#             "error": str(e),
#             "success": False
#         }), 500
#
#
# @app.route('/api/info', methods=['GET'])
# def get_info():
#     """Get API information and capabilities"""
#     return jsonify({
#         "api_name": "RAG API with Gemini",
#         "version": "1.0.0",
#         "description": "Retrieval-Augmented Generation API using Google Gemini and Pinecone",
#         "endpoints": {
#             "/health": "Health check",
#             "/api/chat": "Main chat endpoint for RAG queries",
#             "/api/search": "Search documents in vector store",
#             "/api/status": "System status and configuration",
#             "/api/debug/index": "Debug index statistics",
#             "/api/info": "API information"
#         },
#         "models": {
#             "llm": "gemini-1.5-flash",
#             "embeddings": "models/embedding-001"
#         },
#         "features": [
#             "Advanced reasoning with Gemini",
#             "Document retrieval from Pinecone",
#             "Chat history support",
#             "Multiple fallback strategies",
#             "Enhanced debug information",
#             "LLM-only fallback when no documents found"
#         ]
#     })
#
#
# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({
#         "error": "Endpoint not found",
#         "message": "Please check the API documentation for available endpoints"
#     }), 404
#
#
# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({
#         "error": "Internal server error",
#         "message": "An unexpected error occurred"
#     }), 500
#
#
# if __name__ == '__main__':
#     # Validate environment variables on startup
#     required_env_vars = ['GOOGLE_API_KEY', 'PINECONE_API_KEY']
#     missing_vars = [var for var in required_env_vars if not os.getenv(var)]
#
#     if missing_vars:
#         logger.error(f"Missing required environment variables: {missing_vars}")
#         exit(1)
#
#     logger.info("Starting RAG API server...")
#     logger.info(f"Using index: {INDEX_NAME}")
#
#     # Test connections on startup
#     try:
#         logger.info("Testing Pinecone connection...")
#         pc = get_pinecone_client()
#         indexes = pc.list_indexes()
#         logger.info(f"Available indexes: {[idx.name for idx in indexes]}")
#
#         if INDEX_NAME in [idx.name for idx in indexes]:
#             index = pc.Index(INDEX_NAME)
#             stats = index.describe_index_stats()
#             logger.info(f"Index '{INDEX_NAME}' stats: {stats}")
#
#             if stats.total_vector_count == 0:
#                 logger.warning(
#                     "⚠️  WARNING: Your index exists but contains no vectors! You need to add documents first.")
#         else:
#             logger.error(f"❌ ERROR: Index '{INDEX_NAME}' not found!")
#
#     except Exception as e:
#         logger.error(f"Failed to test Pinecone connection: {e}")
#
#     # Run the Flask app
#     app.run(debug=True, host='0.0.0.0', port=5000)


from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
import hashlib
from typing import Any, Dict, List, Set

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing core functions
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone

# Import constants
try:
    from consts import INDEX_NAME
except ImportError:
    INDEX_NAME = "test1"  # Default fallback

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",  # Development
    "danielsearchai.netlify.app"  # Production
])


# Global variables for caching
llm_instance = None
embeddings_instance = None
vector_store_instance = None
pinecone_client = None


def get_pinecone_client():
    """Get Pinecone client instance"""
    global pinecone_client
    if pinecone_client is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        pinecone_client = Pinecone(api_key=api_key)
    return pinecone_client


def get_gemini_llm():
    """Get Gemini LLM instance with caching"""
    global llm_instance
    if llm_instance is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        llm_instance = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
    return llm_instance


def get_gemini_embeddings():
    """Get Gemini embeddings instance with caching"""
    global embeddings_instance
    if embeddings_instance is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables...")

        embeddings_instance = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",  # Updated to match streamlit version
            google_api_key=api_key
        )
    return embeddings_instance


def initialize_pinecone_retriever():
    """Initialize Pinecone retriever - matches streamlit function"""
    try:
        pc = get_pinecone_client()
        embeddings = get_gemini_embeddings()
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        return None


def retrieve_sources_from_pinecone(prompt: str, k: int = 5, score_threshold: float = 0.7):
    """Retrieve relevant sources from Pinecone - matches streamlit function"""
    vectorstore = initialize_pinecone_retriever()

    if not vectorstore:
        return {"sources": [], "documents": [], "count": 0}

    try:
        # Perform similarity search with scores
        results = vectorstore.similarity_search_with_score(
            query=prompt,
            k=k
        )

        # Filter by score threshold
        filtered_results = [
            (doc, score) for doc, score in results
            if score >= score_threshold
        ]

        if not filtered_results:
            return {"sources": [], "documents": [], "count": 0}

        # Extract sources and organize results
        sources = []
        documents = []

        for doc, score in filtered_results:
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "score": float(score),
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
            documents.append(doc)

        # Remove duplicate sources (keep highest scoring)
        unique_sources = {}
        for source in sources:
            source_url = source["source"]
            if source_url not in unique_sources or source["score"] > unique_sources[source_url]["score"]:
                unique_sources[source_url] = source

        return {
            "sources": list(unique_sources.values()),
            "documents": documents,
            "count": len(filtered_results)
        }

    except Exception as e:
        logger.error(f"Error retrieving from Pinecone: {e}")
        return {"sources": [], "documents": [], "count": 0}


def create_sources_string(source_urls: Set[str]) -> str:
    """Create sources string - matches streamlit function"""
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}. {source}\n"
    return sources_string


def create_enhanced_sources_string(sources_data: list) -> str:
    """Create enhanced sources string with relevance scores - matches streamlit function"""
    if not sources_data:
        return ""

    sources_string = "📚 **Sources utilisées:**\n"
    for i, source in enumerate(sources_data):
        relevance = "🔥" if source["score"] > 0.8 else "⭐" if source["score"] > 0.7 else "📄"
        sources_string += f"{i + 1}. {relevance} [{source['source']}]({source['source']}) (Score: {source['score']:.2f})\n"
    return sources_string


def get_profile_picture_url(email):
    """Get profile picture URL from Gravatar - matches streamlit function"""
    try:
        email_hash = hashlib.md5(email.lower().encode()).hexdigest()
        gravatar_url = f"https://www.gravatar.com/avatar/{email_hash}?d=identicon&s=200"
        return gravatar_url
    except Exception as e:
        logger.error(f"Error generating profile picture URL: {e}")
        return None


def run_gemini_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    """Main LLM function - adapted from streamlit core.py"""
    try:
        vectorstore = initialize_pinecone_retriever()
        if not vectorstore:
            raise ValueError("Failed to initialize Pinecone retriever")

        chat = get_gemini_llm()

        # Get prompts from LangChain hub
        try:
            rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        except Exception as e:
            logger.error(f"Failed to pull prompts from hub: {e}")
            return run_gemini_llm_fallback(query, chat_history)

        # Create chains
        stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

        history_aware_retriever = create_history_aware_retriever(
            llm=chat,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
            prompt=rephrase_prompt
        )

        qa = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=stuff_documents_chain
        )

        result = qa.invoke(input={"input": query, "chat_history": chat_history})
        return result

    except Exception as e:
        logger.error(f"Error in run_gemini_llm: {str(e)}")
        return run_gemini_llm_fallback(query, chat_history)


def run_gemini_llm_fallback(query: str, chat_history: List[Dict[str, Any]] = []):
    """Fallback LLM function"""
    try:
        chat = get_gemini_llm()
        prompt_text = f"""Question: {query}

Please provide a helpful and accurate response to the question above."""

        response = chat.invoke(prompt_text)

        return {
            "answer": response.content if hasattr(response, 'content') else str(response),
            "context": [],
            "source_documents": []
        }
    except Exception as e:
        logger.error(f"Error in fallback: {str(e)}")
        raise


# API ENDPOINTS

@app.route('/api/user-info', methods=['GET'])
def get_user_info():
    """Get user profile information"""
    user_name = "Daniel Glorieux ILBOUDO"
    user_email = "danielglorieuxilboudo@gmail.com"

    return jsonify({
        "name": user_name,
        "email": user_email,
        "profile_picture_url": get_profile_picture_url(user_email)
    })


@app.route('/api/config-status', methods=['GET'])
def get_config_status():
    """Get configuration status - matches streamlit sidebar"""
    gemini_configured = bool(os.getenv("GOOGLE_API_KEY"))
    pinecone_configured = bool(os.getenv("PINECONE_API_KEY"))

    # Test Pinecone connection
    pinecone_connected = False
    pinecone_error = None

    if pinecone_configured:
        try:
            vectorstore = initialize_pinecone_retriever()
            pinecone_connected = vectorstore is not None
        except Exception as e:
            pinecone_error = str(e)

    return jsonify({
        "gemini": {
            "configured": gemini_configured,
            "status": "ready" if gemini_configured else "missing"
        },
        "pinecone": {
            "configured": pinecone_configured,
            "connected": pinecone_connected,
            "status": "ready" if pinecone_connected else "error",
            "error": pinecone_error
        },
        "overall_status": "ready" if (gemini_configured and pinecone_connected) else "error"
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint - matches streamlit processing logic"""
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400

        query = data['query']
        chat_history = data.get('chat_history', [])
        k_results = data.get('k_results', 5)
        score_threshold = data.get('score_threshold', 0.7)

        logger.info(f"Processing query: '{query[:100]}...'")

        # First, retrieve relevant sources from Pinecone (matches streamlit logic)
        retrieval_results = retrieve_sources_from_pinecone(
            query,
            k=k_results,
            score_threshold=score_threshold
        )

        # Call LLM function (matches streamlit logic)
        generated_response = run_gemini_llm(
            query=query,
            chat_history=chat_history
        )

        # Combine sources (matches streamlit logic)
        all_sources = set()

        # Add sources from Pinecone
        for source in retrieval_results["sources"]:
            all_sources.add(source["source"])

        # Add sources from existing context
        if generated_response.get("context"):
            for doc in generated_response["context"]:
                if doc.metadata.get("source"):
                    all_sources.add(doc.metadata["source"])

        # Format response (matches streamlit logic)
        formatted_response = generated_response['answer']

        # Add enhanced sources information
        sources_string = ""
        if retrieval_results["sources"]:
            sources_string = create_enhanced_sources_string(retrieval_results['sources'])
        elif all_sources:
            sources_string = create_sources_string(all_sources)

        if sources_string:
            formatted_response += f"\n\n{sources_string}"

        response = {
            "answer": formatted_response,
            "raw_answer": generated_response['answer'],
            "sources": retrieval_results["sources"],
            "retrieval_count": retrieval_results["count"],
            "success": True,
            "debug_info": {
                "total_sources": len(all_sources),
                "pinecone_sources": len(retrieval_results["sources"]),
                "context_sources": len(generated_response.get("context", []))
            }
        }

        logger.info(f"Response generated with {len(retrieval_results['sources'])} sources")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    """Clear chat history endpoint"""
    return jsonify({
        "message": "Chat history cleared successfully",
        "success": True
    })


@app.route('/api/system-info', methods=['GET'])
def get_system_info():
    """Get system information - matches streamlit info"""
    return jsonify({
        "model_info": {
            "llm": "Gemini Flash + Pinecone RAG",
            "pricing": "Niveau gratuit disponible",
            "speed": "Réponses rapides avec sources"
        },
        "features": [
            "🧠 Model: Gemini Flash + Pinecone RAG",
            "💰 Prix: Niveau gratuit disponible",
            "⚡ Vitesse: Réponses rapides avec sources"
        ],
        "index_name": INDEX_NAME
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Daniel_Glorieux_Search_AI API is running"
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error"
    }), 500


if __name__ == '__main__':
    # Validate environment variables on startup
    required_env_vars = ['GOOGLE_API_KEY', 'PINECONE_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        print("⚠️ Please add the following to your .env file:")
        for var in missing_vars:
            if var == 'GOOGLE_API_KEY':
                print(f"  {var}=your_key_here  # Get from: https://makersuite.google.com/app/apikey")
            elif var == 'PINECONE_API_KEY':
                print(f"  {var}=your_key_here  # Get from: https://app.pinecone.io/")
        exit(1)

    logger.info("🚀 Starting Daniel_Glorieux_Search_AI API server...")
    logger.info(f"📚 Using Pinecone index: {INDEX_NAME}")

    # Test connections on startup
    try:
        logger.info("🔍 Testing Pinecone connection...")
        vectorstore = initialize_pinecone_retriever()
        if vectorstore:
            logger.info("✅ Pinecone connected successfully!")
        else:
            logger.warning("⚠️ Pinecone connection failed!")
    except Exception as e:
        logger.error(f"❌ Pinecone connection error: {e}")

    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)