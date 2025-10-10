import uuid
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from config import MAX_HISTORY_TURNS, MAX_CONTEXT_LENGTH, SESSION_TIMEOUT
import logging_config  # noqa: F401
import logging

# Get logger
logger = logging.getLogger(__name__)


def estimate_token_count(text: str) -> int:
    """
    Rough estimation of token count for context window management.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough approximation: words * 1.3
    words = len(text.split())
    return int(words * 1.3)


def truncate_history_to_fit(history: List[Dict], max_tokens: int) -> List[Dict]:
    """
    Truncate conversation history to fit within token limit.
    
    Args:
        history: List of conversation turns
        max_tokens: Maximum token limit
        
    Returns:
        Truncated history list
    """
    truncated = []
    total_tokens = 0
    
    # Always keep the most recent messages
    for turn in reversed(history):
        turn_tokens = estimate_token_count(turn.get('content', ''))
        if total_tokens + turn_tokens > max_tokens:
            break
        truncated.insert(0, turn)
        total_tokens += turn_tokens
    
    return truncated


class ConversationSession:
    """Represents a single conversation session."""
    
    def __init__(self, session_id: str = None, metadata: Dict = None):
        """
        Initialize a conversation session.
        
        Args:
            session_id: Unique session identifier (generated if not provided)
            metadata: Optional session metadata
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.history = []
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.metadata = metadata or {}
    
    def add_turn(self, role: str, content: str):
        """
        Add a conversation turn to history.
        
        Args:
            role: Either 'user' or 'assistant'
            content: The message content
        """
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_active = datetime.now()
    
    def get_recent_history(self, n: int = None) -> List[Dict]:
        """
        Retrieve the n most recent conversation turns.
        
        Args:
            n: Number of recent turns (uses MAX_HISTORY_TURNS if not specified)
            
        Returns:
            List of recent conversation turns
        """
        n = n or MAX_HISTORY_TURNS * 2  # Each turn has Q and A
        return self.history[-n:] if len(self.history) > n else self.history
    
    def is_expired(self, timeout: int = SESSION_TIMEOUT) -> bool:
        """
        Check if session has expired.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if session has expired
        """
        expiry_time = self.last_active + timedelta(seconds=timeout)
        return datetime.now() > expiry_time
    
    def to_dict(self) -> Dict:
        """
        Serialize session to dictionary.
        
        Returns:
            Dictionary representation of session
        """
        return {
            "session_id": self.session_id,
            "history": self.history,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "metadata": self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'ConversationSession':
        """
        Deserialize session from dictionary.
        
        Args:
            data: Dictionary containing session data
            
        Returns:
            ConversationSession instance
        """
        session = ConversationSession(
            session_id=data.get('session_id'),
            metadata=data.get('metadata', {})
        )
        session.history = data.get('history', [])
        session.created_at = datetime.fromisoformat(data.get('created_at'))
        session.last_active = datetime.fromisoformat(data.get('last_active'))
        return session


class ConversationManager:
    """Manages multiple conversation sessions."""
    
    def __init__(self, storage_path: str = None):
        """
        Initialize conversation manager.
        
        Args:
            storage_path: Optional path for session persistence
        """
        self.sessions = {}  # In-memory storage
        self.storage_path = storage_path or 'logs/sessions'
        
        # Create storage directory if it doesn't exist
        if self.storage_path:
            os.makedirs(self.storage_path, exist_ok=True)
        
        logger.info(f"ConversationManager initialized with storage: {self.storage_path}")
    
    def create_session(self, metadata: Dict = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        session = ConversationSession(metadata=metadata)
        self.sessions[session.session_id] = session
        logger.info(f"Created new session: {session.session_id}")
        return session.session_id
    
    def get_session(self, session_id: str) -> ConversationSession:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationSession instance
            
        Raises:
            ValueError: If session not found or expired
        """
        # Try in-memory first
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.is_expired():
                logger.warning(f"Session {session_id} has expired")
                del self.sessions[session_id]
                raise ValueError(f"Session {session_id} has expired")
            return session
        
        # Try loading from disk
        session = self.load_session(session_id)
        if session:
            if session.is_expired():
                logger.warning(f"Session {session_id} has expired")
                raise ValueError(f"Session {session_id} has expired")
            self.sessions[session_id] = session
            return session
        
        raise ValueError(f"Session {session_id} not found")
    
    def add_message(self, session_id: str, role: str, content: str):
        """
        Add a message to a session.
        
        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        session = self.get_session(session_id)
        session.add_turn(role, content)
        
        # Auto-save if persistence enabled
        if self.storage_path:
            self.save_session(session_id)
    
    def get_context_for_prompt(self, session_id: str, max_turns: int = None) -> List[Dict]:
        """
        Get conversation context for prompt construction.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of turns to include
            
        Returns:
            List of recent conversation turns
        """
        try:
            session = self.get_session(session_id)
            history = session.get_recent_history(max_turns)
            
            # Truncate to fit context window
            history = truncate_history_to_fit(history, MAX_CONTEXT_LENGTH)
            
            return history
        except ValueError as e:
            logger.warning(f"Could not get context: {e}")
            return []
    
    def clear_session(self, session_id: str):
        """
        Clear a session from memory and disk.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Delete from disk
        session_file = os.path.join(self.storage_path, f"session_{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
            logger.info(f"Deleted session file: {session_file}")
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions from memory."""
        expired_ids = []
        for session_id, session in self.sessions.items():
            if session.is_expired():
                expired_ids.append(session_id)
        
        for session_id in expired_ids:
            logger.info(f"Cleaning up expired session: {session_id}")
            del self.sessions[session_id]
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired sessions")
    
    def save_session(self, session_id: str):
        """
        Persist session to disk.
        
        Args:
            session_id: Session identifier
        """
        if not self.storage_path:
            return
        
        try:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            session_file = os.path.join(self.storage_path, f"session_{session_id}.json")
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved session {session_id} to disk")
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Load session from disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationSession if found, None otherwise
        """
        if not self.storage_path:
            return None
        
        try:
            session_file = os.path.join(self.storage_path, f"session_{session_id}.json")
            if not os.path.exists(session_file):
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = ConversationSession.from_dict(data)
            logger.info(f"Loaded session {session_id} from disk")
            return session
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None


if __name__ == '__main__':
    # Test conversation manager
    print("Testing ConversationManager...")
    
    manager = ConversationManager()
    
    # Create session
    session_id = manager.create_session(metadata={"user": "test_user"})
    print(f"\nCreated session: {session_id}")
    
    # Add messages
    manager.add_message(session_id, "user", "What is a writ petition?")
    manager.add_message(session_id, "assistant", "A writ petition is...")
    
    # Get context
    context = manager.get_context_for_prompt(session_id)
    print(f"\nContext ({len(context)} turns):")
    for turn in context:
        print(f"  {turn['role']}: {turn['content'][:50]}...")
    
    # Test session retrieval
    session = manager.get_session(session_id)
    print(f"\nSession active: {not session.is_expired()}")
    print(f"Total turns: {len(session.history)}")
    
    print("\n✅ ConversationManager test complete")
