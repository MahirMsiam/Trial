import sys
import os
from typing import Tuple, Dict
import logging_config  # noqa: F401
from rag_pipeline import RAGPipeline
from conversation_manager import ConversationManager
from config import validate_config

def print_welcome_banner():
    """Display welcome banner and system information."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          BANGLADESH SUPREME COURT RAG CHAT SYSTEM           ║
║                   Legal Research Assistant                   ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print("System Status:")
    
    # Check configuration
    errors = validate_config()
    
    # Check if only FAISS/chunk issues (allows degraded mode)
    has_faiss = os.path.exists("faiss_index.bin") and os.path.exists("chunks_map.json")
    has_db = os.path.exists("extracted_data/database.db")
    
    faiss_only_errors = all("FAISS" in err or "Chunks map" in err or "degraded mode" in err for err in errors) if errors else False
    
    if errors and not has_db:
        # Database missing - critical error, cannot proceed
        print("❌ Critical Configuration Issues:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease fix configuration before proceeding.")
        return False
    elif errors and faiss_only_errors and has_db:
        # Only FAISS/chunks missing - allow degraded mode
        print("⚠️  Configuration Warnings:")
        for error in errors:
            print(f"   - {error}")
        
        print("\n⚠️  Semantic search disabled. Run `python create_index.py` to build the index.")
        print("   → Keyword-only mode available")
        print("\nContinue anyway? (y/n): ", end="")
        
        try:
            choice = input().strip().lower()
            if choice == 'y':
                print("✅ Proceeding in keyword-only mode")
                return True
            else:
                print("❌ Exiting. Please run create_index.py first.")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Exiting.")
            return False
    elif errors:
        # Other errors - show and block
        print("❌ Configuration Issues:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease fix configuration before proceeding.")
        return False
    else:
        print("✅ System configured and ready")
    
    return True


def print_response(response_dict: Dict):
    """Format and display LLM response."""
    print("\n" + "="*60)
    print("ASSISTANT:")
    print("="*60)
    print(response_dict.get('response', 'No response'))
    print()


def print_crime_results(results: Dict):
    """Display crime category search results."""
    print("\n" + "="*60)
    print(f"CRIME CATEGORY: {results.get('crime_type', 'Unknown').upper()}")
    print(f"Found {results.get('count', 0)} cases")
    print("="*60)
    
    cases = results.get('cases', [])
    if cases:
        print("\nCASES:")
        for i, case in enumerate(cases[:10], 1):  # Show first 10
            print(f"\n{i}. {case.get('full_case_id', 'Unknown')}")
            print(f"   Parties: {case.get('petitioner', 'Unknown')} vs {case.get('respondent', 'Unknown')}")
            print(f"   Date: {case.get('judgment_date', 'Unknown')}")
            print(f"   Outcome: {case.get('outcome', 'Not specified')}")
    
    if results.get('summary'):
        print("\n" + "-"*60)
        print("SUMMARY:")
        print("-"*60)
        print(results['summary'])
    print()


def parse_command(user_input: str) -> Tuple[str, str, Dict]:
    """
    Parse user input for special commands.
    
    Returns:
        (command_type, argument, filters_dict)
    """
    if not user_input.startswith('/'):
        return ('query', user_input, {})
    
    parts = user_input.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1] if len(parts) > 1 else ""
    
    return (command, argument, {})


def display_help():
    """Show help information."""
    help_text = """
╔══════════════════════════════════════════════════════════════╗
║                          COMMANDS                            ║
╚══════════════════════════════════════════════════════════════╝

/help           - Show this help message
/new            - Start a new conversation session
/history        - Show current conversation history
/search [type]  - Search for specific crime category
/case [id]      - Get summary of case by ID
/quit, /exit    - Exit the chat

╔══════════════════════════════════════════════════════════════╗
║                       EXAMPLE QUERIES                        ║
╚══════════════════════════════════════════════════════════════╝

"What are the requirements for filing a writ petition?"
"Show me cases about murder from 2023"
"Summarize the judgment in Writ Petition No. 1234 of 2023"
"What laws were cited in corruption cases?"
"Compare cases on Article 102 and Article 103"
"Find judgments about Section 302 of the Penal Code"

╔══════════════════════════════════════════════════════════════╗
║                           TIPS                               ║
╚══════════════════════════════════════════════════════════════╝

- Be specific with case numbers and party names for better results
- Use legal terminology for more accurate retrieval
- Ask follow-up questions to explore topics in depth
- Mention specific years, laws, or sections for filtered results

"""
    print(help_text)


def handle_search_command(pipeline: RAGPipeline, crime_type: str):
    """Execute crime category search."""
    if not crime_type:
        print("Usage: /search [crime_type]")
        print("Examples: /search murder, /search corruption")
        return
    
    print(f"\nSearching for {crime_type} cases...")
    results = pipeline.search_crime_cases(crime_type)
    print_crime_results(results)


def handle_case_command(pipeline: RAGPipeline, case_id: str):
    """Get case summary by ID."""
    if not case_id or not case_id.isdigit():
        print("Usage: /case [case_id]")
        print("Example: /case 12345")
        return
    
    print(f"\nRetrieving case {case_id}...")
    result = pipeline.summarize_case(int(case_id))
    print_response({"response": result.get('summary', 'Case not found')})


def handle_history_command(conversation_manager: ConversationManager, session_id: str):
    """Display conversation history."""
    try:
        history = conversation_manager.get_context_for_prompt(session_id)
        
        if not history:
            print("\nNo conversation history yet.")
            return
        
        print("\n" + "="*60)
        print("CONVERSATION HISTORY:")
        print("="*60)
        
        for i, turn in enumerate(history, 1):
            role = turn.get('role', 'unknown').upper()
            content = turn.get('content', '')
            timestamp = turn.get('timestamp', 'Unknown time')
            
            print(f"\n[{i}] {role} ({timestamp}):")
            print(content[:200] + ("..." if len(content) > 200 else ""))
        
        print()
    except Exception as e:
        print(f"Error retrieving history: {e}")


def interactive_chat():
    """Main interactive chat loop."""
    # Initialize pipeline
    print("\nInitializing RAG system...")
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Create initial session
    session_id = pipeline.conversation_manager.create_session()
    filters = {}
    
    print(f"✅ Chat session started (ID: {session_id[:8]}...)")
    print("\nType /help for commands or start asking questions!")
    print("Type /quit or /exit to end the session.\n")
    
    # Main loop
    while True:
        try:
            # Get user input
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            # Parse for commands
            command_type, argument, _ = parse_command(user_input)
            
            # Handle commands
            if command_type in ['/quit', '/exit']:
                print("\n👋 Goodbye! Session ended.")
                break
            
            elif command_type == '/help':
                display_help()
                continue
            
            elif command_type == '/new':
                session_id = pipeline.conversation_manager.create_session()
                print(f"✅ New session started (ID: {session_id[:8]}...)")
                continue
            
            elif command_type == '/history':
                handle_history_command(pipeline.conversation_manager, session_id)
                continue
            
            elif command_type == '/search':
                handle_search_command(pipeline, argument)
                continue
            
            elif command_type == '/case':
                handle_case_command(pipeline, argument)
                continue
            
            elif command_type.startswith('/'):
                print(f"Unknown command: {command_type}")
                print("Type /help for available commands")
                continue
            
            # Process query
            print("\n🤔 Processing...")
            result = pipeline.process_query(user_input, session_id=session_id, filters=filters)
            
            if result['query_type'] == 'crime_search':
                print_crime_results(result)
            else:
                print_response(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! (Interrupted)")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type /help for assistance.")


def main():
    """Main entry point."""
    if not print_welcome_banner():
        sys.exit(1)
    
    print("\nStarting interactive chat...")
    interactive_chat()


if __name__ == '__main__':
    main()
