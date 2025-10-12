from typing import List, Dict, Optional
from config import SYSTEM_PROMPT_TEMPLATE, CITATION_FORMAT, MAX_HISTORY_TURNS


def build_rag_prompt(query: str, retrieved_contexts: List[Dict], conversation_history: Optional[List[Dict]] = None) -> str:
    """
    Construct the user prompt with retrieved context and conversation history.
    Handles multiple chunks per case (stored in 'chunks' field).
    
    Args:
        query: User's current question
        retrieved_contexts: List of case-level results, each with 'chunks' field containing multiple excerpts
        conversation_history: Optional list of previous conversation turns
        
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    # Add retrieved contexts
    prompt_parts.append("=== RELEVANT JUDGMENT CONTEXTS ===\n")
    
    for i, case_result in enumerate(retrieved_contexts, 1):
        # Check if this is the new format (case-level with chunks) or old format (direct chunks)
        if 'chunks' in case_result:
            # New format: case-level result with multiple chunks
            chunks = case_result['chunks']
            prompt_parts.append(f"Context {i}:")
            prompt_parts.append(f"Case: {case_result.get('full_case_id', 'Unknown')}")
            prompt_parts.append(f"Case Number: {case_result.get('case_number', 'Unknown')}")
            prompt_parts.append(f"Parties: {case_result.get('petitioner', 'Unknown')} vs {case_result.get('respondent', 'Unknown')}")
            prompt_parts.append(f"Date: {case_result.get('judgment_date', 'Unknown')}")
            
            if len(chunks) > 1:
                prompt_parts.append(f"Relevant Excerpts ({len(chunks)} sections):")
                for j, chunk in enumerate(chunks, 1):
                    prompt_parts.append(f"\nSection {j}:")
                    prompt_parts.append(chunk.get('chunk_text', ''))
            else:
                prompt_parts.append("Relevant Text:")
                prompt_parts.append(chunks[0].get('chunk_text', ''))
        else:
            # Old format: direct chunk (backward compatibility)
            prompt_parts.append(f"Context {i}:")
            prompt_parts.append(f"Case: {case_result.get('full_case_id', 'Unknown')}")
            prompt_parts.append(f"Case Number: {case_result.get('case_number', 'Unknown')}")
            prompt_parts.append(f"Parties: {case_result.get('petitioner', 'Unknown')} vs {case_result.get('respondent', 'Unknown')}")
            prompt_parts.append(f"Date: {case_result.get('judgment_date', 'Unknown')}")
            prompt_parts.append("Relevant Text:")
            prompt_parts.append(case_result.get('chunk_text', ''))
        
        prompt_parts.append("---\n")
    
    # Add conversation history if provided
    if conversation_history:
        prompt_parts.append("\n=== PREVIOUS CONVERSATION ===")
        # Limit to recent turns
        recent_history = conversation_history[-MAX_HISTORY_TURNS * 2:]  # Each turn has Q and A
        for turn in recent_history:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            if role == 'user':
                prompt_parts.append(f"Previous Question: {content}")
            else:
                prompt_parts.append(f"Previous Answer: {content[:200]}...")  # Truncate long answers
        prompt_parts.append("")
    
    # Add current query
    prompt_parts.append("\n=== CURRENT QUESTION ===")
    prompt_parts.append(query)
    
    # Add instructions
    prompt_parts.append("\n=== INSTRUCTIONS ===")
    prompt_parts.append("Based on the above contexts from Bangladesh Supreme Court judgments, please answer the current question.")
    prompt_parts.append("Requirements:")
    prompt_parts.append("- Cite specific cases using the format: [Case: <case_number>, Parties: <petitioner> vs <respondent>, Date: <date>]")
    prompt_parts.append("- When multiple sections are provided from the same case, synthesize the information coherently")
    prompt_parts.append("- Only use information from the provided contexts")
    prompt_parts.append("- If the contexts don't contain relevant information, clearly state that")
    prompt_parts.append("- Provide accurate and formal legal language")
    
    return "\n".join(prompt_parts)


def build_crime_category_prompt(query: str, crime_type: str, cases: List[Dict]) -> str:
    """
    Build a specialized prompt for crime category queries.
    
    Args:
        query: Original user query
        crime_type: Detected crime category
        cases: List of cases matching the crime type
        
    Returns:
        Formatted prompt for crime category summary
    """
    prompt_parts = []
    
    prompt_parts.append(f"=== CRIME CATEGORY SEARCH: {crime_type.upper()} ===")
    prompt_parts.append(f"QUERY: {query.upper()}")
    prompt_parts.append(f"Found {len(cases)} cases related to {crime_type}\n")
    
    prompt_parts.append("=== CASE LIST ===")
    for i, case in enumerate(cases, 1):
        prompt_parts.append(f"\n{i}. Case: {case.get('full_case_id', 'Unknown')}")
        prompt_parts.append(f"   Parties: {case.get('petitioner', 'Unknown')} vs {case.get('respondent', 'Unknown')}")
        prompt_parts.append(f"   Date: {case.get('judgment_date', 'Unknown')}")
        prompt_parts.append(f"   Outcome: {case.get('outcome', 'Not specified')}")
        prompt_parts.append(f"   Summary: {case.get('chunk_text', '')[:200]}...")
    
    prompt_parts.append("\n=== INSTRUCTIONS ===")
    prompt_parts.append(f"Please provide a comprehensive analysis of these {crime_type} cases:")
    prompt_parts.append("1. Summarize common patterns and legal issues")
    prompt_parts.append("2. Identify typical outcomes and sentences")
    prompt_parts.append("3. Note any important legal principles or precedents")
    prompt_parts.append("4. Cite specific cases to support your analysis")
    
    return "\n".join(prompt_parts)


def build_comparison_prompt(query: str, cases: List[Dict]) -> str:
    """
    Build a prompt for comparing multiple cases.
    
    Args:
        query: User's comparison query
        cases: List of cases to compare
        
    Returns:
        Formatted comparison prompt
    """
    prompt_parts = []
    
    prompt_parts.append("=== CASE COMPARISON REQUEST ===")
    prompt_parts.append(f"Query: {query}\n")
    
    prompt_parts.append("=== CASES FOR COMPARISON ===")
    for i, case in enumerate(cases, 1):
        prompt_parts.append(f"\nCase {i}: {case.get('full_case_id', 'Unknown')}")
        prompt_parts.append(f"Parties: {case.get('petitioner', 'Unknown')} vs {case.get('respondent', 'Unknown')}")
        prompt_parts.append(f"Date: {case.get('judgment_date', 'Unknown')}")
        prompt_parts.append(f"Content: {case.get('chunk_text', '')}")
        prompt_parts.append(f"Outcome: {case.get('outcome', 'Not specified')}")
        
        # Add laws cited if available
        if 'laws' in case:
            laws = case.get('laws', [])
            if laws:
                laws_list = ', '.join([f"{l['law']} {l['section']}" for l in laws])
                prompt_parts.append(f"Laws Cited: {laws_list}")
        prompt_parts.append("---")
    
    prompt_parts.append("\n=== INSTRUCTIONS ===")
    prompt_parts.append("Compare these cases and provide:")
    prompt_parts.append("1. Key similarities in facts, legal issues, or reasoning")
    prompt_parts.append("2. Important differences in outcomes or judicial interpretation")
    prompt_parts.append("3. Evolution of legal principles (if chronologically relevant)")
    prompt_parts.append("4. A structured comparison table if appropriate")
    
    return "\n".join(prompt_parts)


def build_summarization_prompt(case_data: Dict) -> str:
    """
    Build a prompt for summarizing a specific judgment.
    
    Args:
        case_data: Complete case data dictionary
        
    Returns:
        Formatted summarization prompt
    """
    prompt_parts = []
    
    prompt_parts.append("=== JUDGMENT SUMMARIZATION REQUEST ===\n")
    
    prompt_parts.append("=== CASE DETAILS ===")
    prompt_parts.append(f"Case: {case_data.get('full_case_id', 'Unknown')}")
    prompt_parts.append(f"Case Type: {case_data.get('case_type', 'Unknown')}")
    prompt_parts.append(f"Parties: {case_data.get('petitioner', 'Unknown')} vs {case_data.get('respondent', 'Unknown')}")
    prompt_parts.append(f"Date: {case_data.get('judgment_date', 'Unknown')}")
    prompt_parts.append(f"Judges: {case_data.get('judges', 'Not specified')}")
    prompt_parts.append(f"Outcome: {case_data.get('outcome', 'Not specified')}\n")
    
    # Add advocates if available
    if 'advocates' in case_data and case_data['advocates']:
        prompt_parts.append("Advocates:")
        for adv in case_data['advocates']:
            prompt_parts.append(f"  - {adv.get('name', 'Unknown')} ({adv.get('role', 'Unknown')})")
        prompt_parts.append("")
    
    # Add laws cited
    if 'laws' in case_data and case_data['laws']:
        prompt_parts.append("Laws Cited:")
        for law in case_data['laws']:
            prompt_parts.append(f"  - {law.get('law', 'Unknown')} {law.get('section', '')}")
        prompt_parts.append("")
    
    # Add full text
    prompt_parts.append("=== FULL JUDGMENT TEXT ===")
    prompt_parts.append(case_data.get('full_text', 'No text available'))
    
    prompt_parts.append("\n=== INSTRUCTIONS ===")
    prompt_parts.append("Provide a structured summary of this judgment with the following sections:")
    prompt_parts.append("1. FACTS: Brief overview of the case background and circumstances")
    prompt_parts.append("2. ISSUES: Key legal questions presented to the court")
    prompt_parts.append("3. ARGUMENTS: Summary of arguments by both sides")
    prompt_parts.append("4. HELD: The court's final decision/ruling")
    prompt_parts.append("5. REASONING: Legal reasoning and analysis by the court")
    prompt_parts.append("6. OUTCOME: Practical result and any orders issued")
    prompt_parts.append("\nUse clear headings and maintain formal legal language.")
    
    return "\n".join(prompt_parts)


def format_response_with_citations(response: str, contexts: List[Dict]) -> str:
    """
    Post-process LLM response to ensure proper citation formatting and add source list.
    Injects inline citations for factual statements and appends full source list.
    
    Args:
        response: LLM generated response
        contexts: Retrieved contexts used for generation
        
    Returns:
        Formatted response with inline citations and source list
    """
    if not contexts:
        return response
    
    import re
    
    # Helper to split into sentences
    def split_into_sentences(text: str) -> List[str]:
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    # Map top retrieved contexts to citation strings using CITATION_FORMAT
    citation_strings = []
    seen_cases = set()
    
    for ctx in contexts:
        case_id = ctx.get('full_case_id', 'Unknown')
        if case_id in seen_cases or case_id == 'Unknown':
            continue
        seen_cases.add(case_id)
        
        citation = CITATION_FORMAT.format(
            case_number=ctx.get('case_number', 'Unknown'),
            petitioner=ctx.get('petitioner', 'Unknown'),
            respondent=ctx.get('respondent', 'Unknown'),
            judgment_date=ctx.get('judgment_date', 'Unknown')
        )
        citation_strings.append((case_id, citation, ctx))
    
    if not citation_strings:
        return response
    
    # Inject citations into response
    sentences = split_into_sentences(response)
    
    # Check if response already has bracketed citations
    has_existing_citations = bool(re.search(r'\[Case:', response))
    
    if not has_existing_citations and sentences:
        # Inject citation after first 2-3 substantive sentences
        citation_idx = 0
        injected_count = 0
        max_injections = min(3, len(citation_strings))
        
        for i, sentence in enumerate(sentences):
            # Skip very short sentences (likely headings or fragments)
            if len(sentence) < 30:
                continue
            
            # Inject citation after substantive sentences
            if injected_count < max_injections and citation_idx < len(citation_strings):
                _, citation, _ = citation_strings[citation_idx]
                sentences[i] = sentence + f" {citation}"
                citation_idx += 1
                injected_count += 1
                
                # Space out citations (every 2-3 sentences)
                if injected_count >= max_injections:
                    break
        
        response = ' '.join(sentences)
    
    # Validation: Ensure at least one inline citation exists
    if not re.search(r'\[Case:', response) and citation_strings:
        # Force add citation at end of first paragraph
        first_para_end = response.find('\n\n')
        if first_para_end == -1:
            # Find first sentence end
            first_sentence_end = response.find('. ')
            if first_sentence_end != -1:
                first_para_end = first_sentence_end + 1
        
        if first_para_end != -1:
            _, citation, _ = citation_strings[0]
            response = response[:first_para_end] + f" {citation}" + response[first_para_end:]
        else:
            # Append to end if no good insertion point
            _, citation, _ = citation_strings[0]
            response = response + f" {citation}"
    
    # Append full source list at the end
    formatted_parts = [response]
    formatted_parts.append("\n\n" + "="*60)
    formatted_parts.append("SOURCES:")
    formatted_parts.append("="*60)
    
    for i, (case_id, citation, ctx) in enumerate(citation_strings, 1):
        source_line = f"{i}. {case_id}"
        source_line += f"\n   Parties: {ctx.get('petitioner', 'Unknown')} vs {ctx.get('respondent', 'Unknown')}"
        source_line += f"\n   Date: {ctx.get('judgment_date', 'Unknown')}"
        
        if 'similarity' in ctx:
            source_line += f"\n   Relevance Score: {ctx['similarity']:.2f}"
        elif 'hybrid_score' in ctx:
            source_line += f"\n   Relevance Score: {ctx['hybrid_score']:.2f}"
        
        formatted_parts.append(source_line)
    
    return "\n".join(formatted_parts)


# Fallback response template
FALLBACK_RESPONSE = """I couldn't find relevant information in the judgment database for your query.

Please try:
- Using different keywords or legal terms
- Specifying case numbers, party names, or dates
- Asking about specific legal provisions or case types
- Rephrasing your question with more context

Examples of effective queries:
- "What are the requirements for filing a writ petition under Article 102?"
- "Show me cases about corruption from 2023"
- "What is the legal principle in State vs. [Party Name]?"
- "Find judgments citing Section 302 of the Penal Code"
"""


# Guardrail instructions
GUARDRAIL_PROMPTS = """
IMPORTANT GUARDRAILS:
- Do not provide legal advice or opinions beyond what is stated in the judgments
- Do not speculate about outcomes of hypothetical cases
- Do not discuss legal matters outside Bangladesh jurisdiction
- Do not make predictions about future legal decisions
- Redirect off-topic questions to focus on the judgment database
- Always acknowledge limitations when information is unavailable
"""


def get_system_prompt() -> str:
    """Get the complete system prompt including guardrails."""
    return SYSTEM_PROMPT_TEMPLATE + "\n" + GUARDRAIL_PROMPTS


if __name__ == '__main__':
    # Test prompt building
    print("Testing Prompt Templates...")
    
    test_contexts = [
        {
            "full_case_id": "Writ Petition No. 1234 of 2023",
            "petitioner": "John Doe",
            "respondent": "State",
            "judgment_date": "2023-05-15",
            "chunk_text": "The court held that...",
            "similarity": 0.85
        }
    ]
    
    test_query = "What are the requirements for writ petitions?"
    
    prompt = build_rag_prompt(test_query, test_contexts)
    print("\n" + "="*60)
    print("Generated RAG Prompt:")
    print("="*60)
    print(prompt[:500] + "...")
    
    formatted = format_response_with_citations("Test response", test_contexts)
    print("\n" + "="*60)
    print("Formatted Response with Citations:")
    print("="*60)
    print(formatted)
