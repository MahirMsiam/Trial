"""
Locust Load Testing Script

Load testing for the Legal AI RAG API using Locust.

Usage:
    locust -f load_test_locust.py --host=http://localhost:8000
    
Then open http://localhost:8089 to configure and start the load test.

Load Profiles:
    - Light load: 10 users, 1 user/sec spawn rate
    - Medium load: 50 users, 5 users/sec spawn rate
    - Heavy load: 100+ users, 10 users/sec spawn rate
"""

from locust import HttpUser, task, between, events
import random
import logging

logger = logging.getLogger(__name__)


# Sample data for varied queries
LEGAL_QUESTIONS = [
    "What are the requirements for filing a writ petition?",
    "Explain Section 302 of IPC",
    "What is the procedure for bail application?",
    "What are the grounds for divorce under Indian law?",
    "Explain the concept of natural justice",
    "What is the definition of defamation?",
    "What are the rights of an accused person?",
    "Explain the doctrine of precedent",
    "What is the difference between appeal and revision?",
    "What are the essential elements of a valid contract?",
]

CRIME_QUERIES = [
    "murder cases",
    "theft judgment",
    "corruption cases",
    "assault cases",
    "fraud judgments",
    "robbery cases",
]

KEYWORD_QUERIES = [
    "cases from 2023",
    "High Court Division cases",
    "Supreme Court judgments",
    "criminal appeal",
    "writ petition",
]

SEARCH_FILTERS = [
    {},
    {"case_type": "Criminal Appeal"},
    {"case_year": 2023},
    {"case_type": "Writ Petition", "case_year": 2023},
]

# Sample case IDs (update with actual IDs from your database)
SAMPLE_CASE_IDS = list(range(1, 101))


class SearchUser(HttpUser):
    """
    User that performs various search operations
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    @task(3)
    def keyword_search(self):
        """Perform keyword search (weight: 3)"""
        query = random.choice(KEYWORD_QUERIES)
        filters = random.choice(SEARCH_FILTERS)
        
        with self.client.post(
            "/api/search/keyword",
            json={
                "query": query,
                "limit": 10,
                "filters": filters
            },
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Keyword search failed: {response.status_code}")
    
    @task(2)
    def semantic_search(self):
        """Perform semantic search (weight: 2)"""
        query = random.choice(LEGAL_QUESTIONS)
        
        with self.client.post(
            "/api/search/semantic",
            json={
                "query": query,
                "top_k": 5
            },
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                # FAISS not available
                response.success()  # Don't count as failure
            else:
                response.failure(f"Semantic search failed: {response.status_code}")
    
    @task(2)
    def hybrid_search(self):
        """Perform hybrid search (weight: 2)"""
        query = random.choice(LEGAL_QUESTIONS)
        filters = random.choice(SEARCH_FILTERS)
        
        with self.client.post(
            "/api/search/hybrid",
            json={
                "query": query,
                "top_k": 5,
                "filters": filters
            },
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                # Hybrid search not available
                response.success()
            else:
                response.failure(f"Hybrid search failed: {response.status_code}")
    
    @task(1)
    def crime_search(self):
        """Search crime categories (weight: 1)"""
        query = random.choice(CRIME_QUERIES)
        
        with self.client.post(
            "/api/search/crime",
            json={
                "query": query,
                "limit": 5
            },
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Crime search failed: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Check API health (weight: 1)"""
        with self.client.get("/api/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class ChatUser(HttpUser):
    """
    User that has conversations with the AI
    """
    wait_time = between(2, 5)  # Wait 2-5 seconds between tasks
    
    def on_start(self):
        """Create session when user starts"""
        response = self.client.post("/api/session")
        if response.status_code == 200:
            self.session_id = response.json().get('session_id')
            logger.info(f"Created session: {self.session_id}")
        else:
            self.session_id = None
            logger.error("Failed to create session")
    
    @task(5)
    def ask_question(self):
        """Ask legal question (weight: 5)"""
        if not self.session_id:
            return
        
        query = random.choice(LEGAL_QUESTIONS)
        filters = random.choice(SEARCH_FILTERS)
        
        with self.client.post(
            "/api/chat",
            json={
                "query": query,
                "session_id": self.session_id,
                "filters": filters
            },
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Chat query failed: {response.status_code}")
    
    @task(1)
    def get_history(self):
        """Retrieve conversation history (weight: 1)"""
        if not self.session_id:
            return
        
        with self.client.get(
            f"/api/session/{self.session_id}/history",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Session might have expired
                response.success()
            else:
                response.failure(f"Get history failed: {response.status_code}")
    
    def on_stop(self):
        """Delete session when user stops"""
        if self.session_id:
            self.client.delete(f"/api/session/{self.session_id}")
            logger.info(f"Deleted session: {self.session_id}")


class CaseUser(HttpUser):
    """
    User that explores specific cases
    """
    wait_time = between(1, 4)  # Wait 1-4 seconds between tasks
    
    @task(3)
    def get_case_summary(self):
        """Get case summary (weight: 3)"""
        case_id = random.choice(SAMPLE_CASE_IDS)
        
        with self.client.post(
            f"/api/case/{case_id}/summary",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Case not found, don't count as failure
                response.success()
            else:
                response.failure(f"Case summary failed: {response.status_code}")
    
    @task(1)
    def compare_cases(self):
        """Compare multiple cases (weight: 1)"""
        # Select 2-3 random case IDs
        num_cases = random.randint(2, 3)
        case_ids = random.sample(SAMPLE_CASE_IDS, num_cases)
        
        with self.client.post(
            "/api/cases/compare",
            json={"case_ids": case_ids},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # One or more cases not found
                response.success()
            else:
                response.failure(f"Compare cases failed: {response.status_code}")


class MixedUser(HttpUser):
    """
    Mixed user behavior - combines all user types
    Represents realistic user behavior
    """
    wait_time = between(1, 5)
    
    def on_start(self):
        """Create session when user starts"""
        response = self.client.post("/api/session")
        if response.status_code == 200:
            self.session_id = response.json().get('session_id')
        else:
            self.session_id = None
    
    @task(3)
    def search_operation(self):
        """Perform search (weight: 3)"""
        search_type = random.choice(['keyword', 'semantic', 'hybrid'])
        query = random.choice(LEGAL_QUESTIONS + KEYWORD_QUERIES)
        
        if search_type == 'keyword':
            self.client.post(
                "/api/search/keyword",
                json={"query": query, "limit": 10, "filters": {}}
            )
        elif search_type == 'semantic':
            self.client.post(
                "/api/search/semantic",
                json={"query": query, "top_k": 5}
            )
        else:  # hybrid
            self.client.post(
                "/api/search/hybrid",
                json={"query": query, "top_k": 5, "filters": {}}
            )
    
    @task(2)
    def chat_operation(self):
        """Chat with AI (weight: 2)"""
        if not self.session_id:
            return
        
        query = random.choice(LEGAL_QUESTIONS)
        self.client.post(
            "/api/chat",
            json={"query": query, "session_id": self.session_id, "filters": {}}
        )
    
    @task(1)
    def case_operation(self):
        """Explore cases (weight: 1)"""
        case_id = random.choice(SAMPLE_CASE_IDS)
        self.client.post(f"/api/case/{case_id}/summary")
    
    def on_stop(self):
        """Cleanup on stop"""
        if self.session_id:
            self.client.delete(f"/api/session/{self.session_id}")


# Event handlers for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    logger.info("Load test starting...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    logger.info("Load test completed!")
    
    # Print summary statistics
    stats = environment.stats
    logger.info(f"Total requests: {stats.total.num_requests}")
    logger.info(f"Total failures: {stats.total.num_failures}")
    logger.info(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"RPS: {stats.total.current_rps:.2f}")


if __name__ == "__main__":
    import os
    
    print("=" * 60)
    print("Legal AI RAG API - Load Testing with Locust")
    print("=" * 60)
    print()
    print("Usage:")
    print("  locust -f load_test_locust.py --host=http://localhost:8000")
    print()
    print("Then open http://localhost:8089 in your browser")
    print()
    print("Recommended Load Profiles:")
    print("  Light:  10 users, 1 user/sec spawn rate")
    print("  Medium: 50 users, 5 users/sec spawn rate")
    print("  Heavy:  100 users, 10 users/sec spawn rate")
    print()
    print("Available User Classes:")
    print("  - SearchUser: Performs various search operations")
    print("  - ChatUser: Has conversations with the AI")
    print("  - CaseUser: Explores specific cases")
    print("  - MixedUser: Combination of all behaviors (default)")
    print()
    print("To use a specific user class:")
    print("  locust -f load_test_locust.py --host=http://localhost:8000 SearchUser")
    print()
    print("=" * 60)
