"""
Enhanced RAG (Retrieval Augmented Generation) Layer
Manages compliance policy retrieval and semantic search.
Architecture Decision: Separate RAG into its own layer to:
  1. Support both FAISS (local) and Pinecone (cloud) backends
  2. Implement caching for frequently accessed policies
  3. Enable real-time policy updates
  4. Track policy relevance scores
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class PolicyDocument:
    """Represents a compliance policy document"""
    id: str
    title: str
    content: str
    category: str
    severity: str  # critical, high, medium, low
    created_at: datetime
    embedding: Optional[np.ndarray] = None
    relevance_score: float = 0.0


class EmbeddingProvider:
    """Handles text embedding using sentence-transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding model"""
        self.model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded: {model_name}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Convert text to embedding"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts efficiently"""
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32)
    
    def similarity_score(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))


class LocalPolicyStore:
    """In-memory policy store with FAISS-like functionality"""
    
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.policies: Dict[str, PolicyDocument] = {}
        self.embeddings = embedding_provider
        self.policy_embeddings: Dict[str, np.ndarray] = {}
    
    def add_policy(self, policy: PolicyDocument) -> None:
        """Add policy to store"""
        policy.embedding = self.embeddings.embed_text(policy.content)
        self.policies[policy.id] = policy
        self.policy_embeddings[policy.id] = policy.embedding
        logger.info(f"Policy added: {policy.title}")
    
    def remove_policy(self, policy_id: str) -> None:
        """Remove policy from store"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            del self.policy_embeddings[policy_id]
            logger.info(f"Policy removed: {policy_id}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[PolicyDocument, float]]:
        """Search for relevant policies"""
        if not self.policies:
            return []
        
        query_embedding = self.embeddings.embed_text(query)
        scores = {}
        
        for policy_id, policy_embedding in self.policy_embeddings.items():
            score = self.embeddings.similarity_score(query_embedding, policy_embedding)
            scores[policy_id] = score
        
        # Return top-k policies sorted by relevance
        top_policies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.policies[pid], score) for pid, score in top_policies]
    
    def get_by_category(self, category: str) -> List[PolicyDocument]:
        """Get all policies in a category"""
        return [p for p in self.policies.values() if p.category == category]


class ComplianceRAGEnhanced:
    """Enterprise-grade RAG system for compliance checking"""
    
    def __init__(self, use_local_store: bool = True):
        """Initialize RAG system"""
        self.embedding_provider = EmbeddingProvider()
        self.policy_store = LocalPolicyStore(self.embedding_provider) if use_local_store else None
        self.query_cache: Dict[str, List[PolicyDocument]] = {}
        self._load_default_policies()
    
    def _load_default_policies(self) -> None:
        """Load default policies from policy.txt if exists"""
        policy_file = os.path.join(os.path.dirname(__file__), "..", "policy.txt")
        
        if not os.path.exists(policy_file):
            logger.warning("policy.txt not found, skipping default policy load")
            return
        
        try:
            with open(policy_file, "r") as f:
                content = f.read()
            
            # Parse policies (simple format: title | category | severity | content)
            policies = content.split("---")
            for i, policy_text in enumerate(policies):
                lines = policy_text.strip().split("\n", 3)
                if len(lines) >= 4:
                    policy = PolicyDocument(
                        id=f"policy_{i}",
                        title=lines[0].strip(),
                        category=lines[1].strip(),
                        severity=lines[2].strip(),
                        content=lines[3].strip(),
                        created_at=datetime.now()
                    )
                    self.policy_store.add_policy(policy)
            
            logger.info(f"Loaded {len(policies)} policies from policy.txt")
        except Exception as e:
            logger.error(f"Error loading policies: {e}")
    
    def get_rules_for_context(self, context: str, top_k: int = 3) -> str:
        """
        Get most relevant compliance rules for given context.
        Returns formatted string of relevant policies.
        """
        cache_key = hash(context) % 100000
        
        if cache_key in self.query_cache:
            results = self.query_cache[cache_key]
        else:
            results, scores = self.search_policies(context, top_k)
            self.query_cache[cache_key] = results
        
        if not results:
            return "No relevant policies found. Default: Follow customer protection regulations."
        
        formatted = "RELEVANT COMPLIANCE POLICIES:\n" + "-" * 50 + "\n"
        for i, policy in enumerate(results, 1):
            formatted += f"\n{i}. {policy.title} [Severity: {policy.severity}]\n"
            formatted += f"   {policy.content}\n"
        
        return formatted
    
    def search_policies(self, query: str, top_k: int = 5) -> Tuple[List[PolicyDocument], List[float]]:
        """
        Search policies with relevance scores.
        Returns (policies, scores)
        """
        if not self.policy_store:
            return [], []
        
        results = self.policy_store.search(query, top_k)
        policies = [p for p, _ in results]
        scores = [s for _, s in results]
        
        logger.debug(f"Policy search returned {len(policies)} results")
        return policies, scores
    
    def validate_compliance(self, text: str, strategy: str = "strict") -> Dict[str, Any]:
        """
        Validate text against stored policies.
        Strategy: 'strict' (match severity), 'lenient' (key terms only)
        """
        policies, scores = self.search_policies(text, top_k=10)
        
        violations = []
        for policy, score in zip(policies, scores):
            if strategy == "strict" and score > 0.6:
                violations.append({
                    "policy": policy.title,
                    "severity": policy.severity,
                    "relevance_score": float(score),
                    "guidance": policy.content
                })
            elif strategy == "lenient" and score > 0.4:
                violations.append({
                    "policy": policy.title,
                    "severity": policy.severity,
                    "relevance_score": float(score),
                    "guidance": policy.content
                })
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "policy_check_count": len(policies),
            "avg_relevance_score": float(np.mean(scores)) if scores else 0.0
        }
    
    def add_custom_policy(self, title: str, content: str, category: str, severity: str) -> str:
        """Add custom policy at runtime"""
        policy_id = f"custom_{datetime.now().timestamp()}"
        policy = PolicyDocument(
            id=policy_id,
            title=title,
            content=content,
            category=category,
            severity=severity,
            created_at=datetime.now()
        )
        self.policy_store.add_policy(policy)
        self.query_cache.clear()  # Clear cache when policies change
        return policy_id
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of loaded policies"""
        if not self.policy_store:
            return {}
        
        policies = list(self.policy_store.policies.values())
        categories = {}
        severities = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for policy in policies:
            if policy.category not in categories:
                categories[policy.category] = 0
            categories[policy.category] += 1
            if policy.severity in severities:
                severities[policy.severity] += 1
        
        return {
            "total_policies": len(policies),
            "by_category": categories,
            "by_severity": severities
        }


# Backward compatibility with existing code
class ComplianceRAG(ComplianceRAGEnhanced):
    """Backward compatible class name"""
    pass
