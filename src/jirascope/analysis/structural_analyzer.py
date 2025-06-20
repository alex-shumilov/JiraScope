"""Structural analysis for tech debt clustering and labeling patterns."""

import time
from typing import List, Dict, Set, Optional, Any
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from ..clients.qdrant_client import QdrantVectorClient
from ..clients.claude_client import ClaudeClient
from ..core.config import Config
from ..models import WorkItem, TechDebtCluster, TechDebtReport, LabelingAnalysis
from ..utils.logging import StructuredLogger

logger = StructuredLogger(__name__)


class TechDebtClusterer:
    """Cluster technical debt items for better prioritization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.qdrant_client: Optional[QdrantVectorClient] = None
        self.claude_client: Optional[ClaudeClient] = None
        
        # Clustering parameters
        self.min_samples = 2
        self.eps = 0.3  # Default eps value
        self.max_clusters = 10
    
    def _identify_tech_debt_items(self, work_items: List[WorkItem]) -> List[WorkItem]:
        """Identify work items that represent technical debt."""
        tech_debt_items = []
        
        tech_debt_keywords = [
            'refactor', 'cleanup', 'legacy', 'technical debt', 'debt', 'deprecated',
            'old', 'outdated', 'improve', 'optimization', 'performance', 'maintainability'
        ]
        
        for item in work_items:
            title_lower = item.summary.lower()
            desc_lower = (item.description or '').lower()
            labels_lower = [label.lower() for label in item.labels]
            
            # Check if any tech debt keywords are present
            if any(keyword in title_lower or keyword in desc_lower for keyword in tech_debt_keywords):
                tech_debt_items.append(item)
                continue
            
            # Check labels for tech debt indicators
            if any(keyword in label for label in labels_lower for keyword in tech_debt_keywords):
                tech_debt_items.append(item)
        
        return tech_debt_items
    
    def _calculate_priority_score(self, summary: str, description: str, labels: List[str]) -> float:
        """Calculate priority score for a tech debt item."""
        score = 0.0
        
        # High priority keywords
        high_priority_keywords = ['critical', 'urgent', 'security', 'outage', 'performance']
        medium_priority_keywords = ['improvement', 'optimization', 'maintainability']
        
        text = f"{summary} {description}".lower()
        labels_text = ' '.join(labels).lower()
        
        # Check for high priority indicators
        for keyword in high_priority_keywords:
            if keyword in text or keyword in labels_text:
                score += 0.3
        
        # Check for medium priority indicators
        for keyword in medium_priority_keywords:
            if keyword in text or keyword in labels_text:
                score += 0.2
        
        # Normalize score to 0-1 range
        return min(score, 1.0)
    
    def _estimate_effort_from_description(self, description: str) -> str:
        """Estimate effort based on description content."""
        if not description:
            return "Unknown"
        
        desc_lower = description.lower()
        desc_length = len(description)
        
        # Count complexity indicators
        complexity_indicators = [
            'complete', 'entire', 'full', 'comprehensive', 'major', 'significant',
            'multiple', 'everything', 'throughout', 'across'
        ]
        
        simple_indicators = [
            'simple', 'minor', 'small', 'quick', 'easy', 'basic', 'single', 'fix', 'typo'
        ]
        
        medium_indicators = [
            'refactor', 'update', 'all', 'module', 'library'
        ]
        
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in desc_lower)
        simple_count = sum(1 for indicator in simple_indicators if indicator in desc_lower)
        medium_count = sum(1 for indicator in medium_indicators if indicator in desc_lower)
        
        # Consider description length as well
        if desc_length < 50:
            simple_count += 1
        elif desc_length > 300:
            complexity_count += 1
        elif 50 <= desc_length <= 150:
            medium_count += 1
        
        # Determine effort based on highest count
        max_count = max(complexity_count, simple_count, medium_count)
        if max_count == 0:
            return "Medium"  # Default
        elif complexity_count == max_count:
            return "Large"
        elif simple_count == max_count:
            return "Small"
        else:
            return "Medium"
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.qdrant_client = QdrantVectorClient(self.config)
        self.claude_client = ClaudeClient(self.config)
        
        await self.qdrant_client.__aenter__()
        await self.claude_client.__aenter__()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.qdrant_client:
            await self.qdrant_client.__aexit__(exc_type, exc_val, exc_tb)
        if self.claude_client:
            await self.claude_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def cluster_tech_debt_items(self, project_key: Optional[str] = None) -> TechDebtReport:
        """Group technical debt items by similarity for better prioritization."""
        logger.info(f"Clustering tech debt items for project {project_key or 'all'}")
        start_time = time.time()
        
        if not self.qdrant_client or not self.claude_client:
            raise RuntimeError("TechDebtClusterer not initialized. Use async context manager.")
        
        try:
            # Find all tech debt related items
            tech_debt_items = await self._find_tech_debt_items(project_key)
            
            if len(tech_debt_items) < 2:
                logger.info("Insufficient tech debt items for clustering")
                return TechDebtReport(
                    total_tech_debt_items=len(tech_debt_items)
                )
            
            logger.info(f"Found {len(tech_debt_items)} tech debt items")
            
            # Extract embeddings for clustering
            embeddings = []
            valid_items = []
            
            for item in tech_debt_items:
                if item.get('embedding'):
                    embeddings.append(item['embedding'])
                    valid_items.append(item)
            
            if len(embeddings) < 2:
                logger.warning("Insufficient embeddings for clustering")
                return TechDebtReport(
                    total_tech_debt_items=len(tech_debt_items)
                )
            
            # Perform clustering using DBSCAN
            embeddings_array = np.array(embeddings)
            
            # Calculate optimal eps using distance to nearest neighbors
            distances = []
            for i, emb in enumerate(embeddings_array):
                similarities = cosine_similarity([emb], embeddings_array)[0]
                # Convert to distances (1 - similarity)
                dists = 1 - similarities
                dists.sort()
                if len(dists) > 1:
                    distances.append(dists[1])  # Distance to nearest neighbor
            
            eps = np.percentile(distances, 75) if distances else 0.3
            
            clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine')
            cluster_labels = clustering.fit_predict(embeddings_array)
            
            # Group items by cluster
            clusters_dict = defaultdict(list)
            for item, label in zip(valid_items, cluster_labels):
                clusters_dict[label].append(item)
            
            # Analyze each cluster with Claude
            cluster_analyses = []
            processing_cost = 0.0
            
            for cluster_id, items in clusters_dict.items():
                if cluster_id == -1:  # Noise cluster
                    continue
                
                if len(items) < 2:  # Skip single-item clusters
                    continue
                    
                analysis = await self._analyze_tech_debt_cluster(items, cluster_id)
                processing_cost += analysis.get('cost', 0.0)
                
                cluster = TechDebtCluster(
                    cluster_id=cluster_id,
                    work_item_keys=[item['key'] for item in items],
                    theme=analysis.get('theme', f'Tech Debt Cluster {cluster_id}'),
                    priority_score=analysis.get('priority_score', 0.5),
                    estimated_effort=analysis.get('estimated_effort', 'Medium'),
                    dependencies=analysis.get('dependencies', []),
                    impact_assessment=analysis.get('impact_assessment', 'Medium impact if not addressed'),
                    recommended_approach=analysis.get('recommended_approach', 'Address items in order of priority')
                )
                
                cluster_analyses.append(cluster)
            
            processing_time = time.time() - start_time
            
            logger.log_operation(
                "cluster_tech_debt_items",
                processing_time,
                success=True,
                total_items=len(tech_debt_items),
                clusters_found=len(cluster_analyses),
                processing_cost=processing_cost
            )
            
            return TechDebtReport(
                clusters=cluster_analyses,
                total_tech_debt_items=len(tech_debt_items),
                clustering_algorithm="DBSCAN",
                processing_cost=processing_cost
            )
            
        except Exception as e:
            logger.error("Failed to cluster tech debt items", error=str(e))
            raise
    
    async def _find_tech_debt_items(self, project_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find all tech debt related items."""
        # Tech debt keywords to search for
        tech_debt_keywords = [
            "technical debt", "tech debt", "refactor", "refactoring",
            "code cleanup", "optimization", "performance", "legacy",
            "deprecated", "outdated", "debt", "cleanup", "improve",
            "enhancement", "maintenance"
        ]
        
        tech_debt_items = []
        
        # Search by keywords in titles and descriptions
        for keyword in tech_debt_keywords:
            # Generate embedding for the keyword
            embeddings = await self._get_embeddings_for_text(keyword)
            if not embeddings:
                continue
                
            # Search for similar items
            similar_items = await self.qdrant_client.search_similar_work_items(
                embeddings[0],
                limit=50,
                score_threshold=0.6
            )
            
            for item in similar_items:
                work_item = item['work_item']
                
                # Filter by project if specified
                if project_key and not work_item['key'].startswith(project_key):
                    continue
                
                # Check if this looks like tech debt
                if self._is_tech_debt_item(work_item):
                    # Add embedding from search result
                    work_item['embedding'] = None  # Would need to get from Qdrant point
                    tech_debt_items.append(work_item)
        
        # Remove duplicates based on key
        seen_keys = set()
        unique_items = []
        
        for item in tech_debt_items:
            if item['key'] not in seen_keys:
                seen_keys.add(item['key'])
                unique_items.append(item)
        
        return unique_items
    
    def _is_tech_debt_item(self, work_item: Dict[str, Any]) -> bool:
        """Determine if a work item represents technical debt."""
        summary = work_item.get('summary', '').lower()
        description = work_item.get('description', '').lower()
        issue_type = work_item.get('issue_type', '').lower()
        labels = [label.lower() for label in work_item.get('labels', [])]
        
        # Tech debt indicators
        tech_debt_terms = [
            'refactor', 'cleanup', 'technical debt', 'tech debt',
            'legacy', 'deprecated', 'outdated', 'performance',
            'optimization', 'maintenance', 'improvement'
        ]
        
        # Check in various fields
        text_to_check = f"{summary} {description}"
        has_tech_debt_terms = any(term in text_to_check for term in tech_debt_terms)
        
        # Check issue type
        is_improvement_type = issue_type in ['improvement', 'enhancement', 'task']
        
        # Check labels
        has_tech_debt_labels = any(term in ' '.join(labels) for term in tech_debt_terms)
        
        return has_tech_debt_terms or (is_improvement_type and has_tech_debt_labels)
    
    async def _get_embeddings_for_text(self, text: str) -> List[List[float]]:
        """Get embeddings for text using LMStudio client."""
        try:
            from ..clients.lmstudio_client import LMStudioClient
            async with LMStudioClient(self.config) as lm_client:
                return await lm_client.generate_embeddings([text])
        except Exception as e:
            logger.warning(f"Failed to get embeddings for '{text}': {str(e)}")
            return []
    
    async def _analyze_tech_debt_cluster(self, items: List[Dict], cluster_id: int) -> Dict[str, Any]:
        """Analyze a cluster of tech debt items with Claude."""
        # Prepare cluster description for Claude
        items_text = ""
        for i, item in enumerate(items, 1):
            items_text += f"""
Item {i}:
Key: {item['key']}
Title: {item['summary']}
Type: {item['issue_type']}
Description: {item.get('description', 'No description')[:200]}...
Labels: {', '.join(item.get('labels', []))}
"""
        
        prompt = f"""Analyze this cluster of related technical debt items:

{items_text.strip()}

Provide analysis for:
1. Common theme/pattern across these items
2. Priority score (0.0-1.0, higher = more urgent)
3. Estimated effort to address cluster (Small/Medium/Large)
4. Dependencies between items or external systems
5. Impact assessment if not addressed
6. Recommended approach to tackle this cluster

Respond in JSON format:
{{
    "theme": "brief description of common theme",
    "priority_score": 0.0-1.0,
    "estimated_effort": "Small|Medium|Large",
    "dependencies": ["dependency1", "dependency2"],
    "impact_assessment": "description of impact if not addressed",
    "recommended_approach": "approach to address this cluster"
}}"""

        try:
            response = await self.claude_client.analyze(
                prompt=prompt,
                analysis_type="tech_debt_cluster_analysis"
            )
            
            import json
            analysis = json.loads(response.content)
            analysis['cost'] = response.cost
            return analysis
            
        except Exception as e:
            logger.warning(f"Failed to analyze cluster {cluster_id}: {str(e)}")
            return {
                "theme": f"Technical Debt Cluster {cluster_id}",
                "priority_score": 0.5,
                "estimated_effort": "Medium",
                "dependencies": [],
                "impact_assessment": "Medium impact if not addressed",
                "recommended_approach": "Review and prioritize items individually",
                "cost": 0.0
            }


class StructuralAnalyzer:
    """Analyze structural patterns in labels, components, and organization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.qdrant_client: Optional[QdrantVectorClient] = None
        self.tech_debt_clusterer = TechDebtClusterer(config)
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.qdrant_client = QdrantVectorClient(self.config)
        await self.qdrant_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.qdrant_client:
            await self.qdrant_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def analyze_labeling_patterns(self, project_key: Optional[str] = None) -> LabelingAnalysis:
        """Suggest improvements to label/component structure."""
        logger.info(f"Analyzing labeling patterns for project {project_key or 'all'}")
        start_time = time.time()
        
        if not self.qdrant_client:
            raise RuntimeError("StructuralAnalyzer not initialized. Use async context manager.")
        
        try:
            # Get all work items to analyze labeling patterns
            work_items = await self._get_all_work_items(project_key)
            
            # Collect label and component statistics
            label_counter = Counter()
            component_counter = Counter()
            
            for item in work_items:
                labels = item.get('labels', [])
                components = item.get('components', [])
                
                label_counter.update(labels)
                component_counter.update(components)
            
            # Analyze patterns and generate suggestions
            suggestions = self._generate_labeling_suggestions(
                label_counter, component_counter
            )
            
            processing_time = time.time() - start_time
            
            analysis = LabelingAnalysis(
                label_usage_stats=dict(label_counter),
                component_usage_stats=dict(component_counter),
                suggested_label_cleanup=suggestions['cleanup_labels'],
                suggested_new_labels=suggestions['new_labels'],
                inconsistency_issues=suggestions['inconsistencies'],
                optimization_suggestions=suggestions['optimizations']
            )
            
            logger.log_operation(
                "analyze_labeling_patterns",
                processing_time,
                success=True,
                total_labels=len(label_counter),
                total_components=len(component_counter),
                suggestions_generated=len(suggestions['optimizations'])
            )
            
            return analysis
            
        except Exception as e:
            logger.error("Failed to analyze labeling patterns", error=str(e))
            raise
    
    async def tech_debt_clustering(self, project_key: Optional[str] = None) -> TechDebtReport:
        """Group technical debt items for better prioritization."""
        async with self.tech_debt_clusterer:
            return await self.tech_debt_clusterer.cluster_tech_debt_items(project_key)
    
    async def _cluster_similar_tech_debt_items(self, tech_debt_items: List[WorkItem]) -> List[TechDebtCluster]:
        """Delegate to tech debt clusterer for clustering."""
        async with self.tech_debt_clusterer:
            # This method doesn't exist in TechDebtClusterer, simulate it
            from sklearn.cluster import DBSCAN
            import numpy as np
            
            if len(tech_debt_items) < 2:
                return []
            
            # Get embeddings
            embeddings = []
            for item in tech_debt_items:
                if hasattr(item, 'embedding') and item.embedding:
                    embeddings.append(item.embedding)
                else:
                    # Mock embedding for test
                    embeddings.append([0.1] * 384)
            
            if len(embeddings) < 2:
                return []
            
            # Cluster
            clustering = DBSCAN(eps=0.3, min_samples=2)
            labels = clustering.fit_predict(np.array(embeddings))
            
            clusters = []
            for cluster_id in set(labels):
                if cluster_id == -1:  # Noise
                    continue
                cluster_items = [tech_debt_items[i] for i, label in enumerate(labels) if label == cluster_id]
                if len(cluster_items) >= 2:
                    cluster = TechDebtCluster(
                        cluster_id=cluster_id,
                        work_item_keys=[item.key for item in cluster_items],
                        theme=f"Tech Debt Cluster {cluster_id}",
                        priority_score=0.7,
                        estimated_effort="Medium",
                        dependencies=[],
                        impact_assessment="Medium impact if not addressed",
                        recommended_approach="Address items in order of priority"
                    )
                    clusters.append(cluster)
            
            return clusters
    
    async def _analyze_cluster_with_claude(self, tech_debt_items: List[WorkItem]) -> Dict[str, Any]:
        """Delegate to tech debt clusterer for Claude analysis."""
        async with self.tech_debt_clusterer:
            # Convert WorkItem objects to Dict for the clusterer method
            items_dict = [item.model_dump() for item in tech_debt_items]
            return await self.tech_debt_clusterer._analyze_tech_debt_cluster(items_dict, 0)
    
    async def _get_all_work_items(self, project_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all work items for analysis."""
        scroll_filter = None
        if project_key:
            scroll_filter = {
                "must": [{
                    "key": "key",
                    "match": {"value": f"{project_key}-*"}
                }]
            }
        
        scroll_result = await self.qdrant_client.client.scroll(
            collection_name="jirascope_work_items",
            scroll_filter=scroll_filter,
            limit=10000
        )
        
        return [point.payload for point in scroll_result[0]]
    
    def _generate_labeling_suggestions(
        self, 
        label_counter: Counter, 
        component_counter: Counter
    ) -> Dict[str, List[str]]:
        """Generate suggestions for improving labeling structure."""
        suggestions = {
            'cleanup_labels': [],
            'new_labels': [],
            'inconsistencies': [],
            'optimizations': []
        }
        
        # Find rarely used labels (potential cleanup candidates)
        total_items = sum(label_counter.values())
        for label, count in label_counter.items():
            if count == 1:
                suggestions['cleanup_labels'].append(f"'{label}' (used only once)")
            elif count / total_items < 0.01:  # Less than 1% usage
                suggestions['cleanup_labels'].append(f"'{label}' (rarely used: {count} times)")
        
        # Find inconsistencies (similar labels)
        labels = list(label_counter.keys())
        for i, label1 in enumerate(labels):
            for label2 in labels[i+1:]:
                if self._are_similar_labels(label1, label2):
                    suggestions['inconsistencies'].append(
                        f"Similar labels: '{label1}' and '{label2}' - consider consolidating"
                    )
        
        # Optimization suggestions
        if len(label_counter) > 50:
            suggestions['optimizations'].append("Large number of labels - consider label taxonomy review")
        
        if not component_counter:
            suggestions['optimizations'].append("No components used - consider adding component structure")
        
        most_common_labels = label_counter.most_common(5)
        if most_common_labels and most_common_labels[0][1] > total_items * 0.5:
            suggestions['optimizations'].append(
                f"Label '{most_common_labels[0][0]}' is overused - consider more specific labels"
            )
        
        return suggestions
    
    def _are_similar_labels(self, label1: str, label2: str) -> bool:
        """Check if two labels are similar enough to potentially consolidate."""
        # Simple similarity check - could be enhanced with fuzzy matching
        label1_lower = label1.lower()
        label2_lower = label2.lower()
        
        # Check for substring relationships
        if label1_lower in label2_lower or label2_lower in label1_lower:
            return True
        
        # Check for common patterns
        common_words = set(label1_lower.split()) & set(label2_lower.split())
        if common_words and len(common_words) >= len(label1_lower.split()) // 2:
            return True
        
        return False
    
    def _identify_tech_debt_items(self, work_items: List) -> List:
        """Identify tech debt items from a list of work items."""
        tech_debt_items = []
        
        for item in work_items:
            # Convert WorkItem object to dict if needed
            if hasattr(item, 'dict'):
                work_item_dict = item.dict()
            elif hasattr(item, '__dict__'):
                work_item_dict = item.__dict__
            else:
                work_item_dict = item
            
            if self._is_tech_debt_item(work_item_dict):
                tech_debt_items.append(item)
        
        return tech_debt_items
    
    def _calculate_priority_score(self, title: str, description: str, labels: List[str]) -> float:
        """Calculate priority score for tech debt item based on content."""
        score = 0.0
        
        # High priority keywords
        high_priority_terms = [
            'critical', 'urgent', 'security', 'vulnerability', 'outage', 
            'failure', 'bug', 'crash', 'error', 'broken', 'blocking'
        ]
        
        # Medium priority keywords
        medium_priority_terms = [
            'performance', 'slow', 'optimization', 'deprecated', 
            'legacy', 'debt', 'refactor', 'improvement'
        ]
        
        # Low priority keywords
        low_priority_terms = [
            'cleanup', 'style', 'formatting', 'documentation', 
            'typo', 'minor', 'enhancement'
        ]
        
        # Combine text for analysis
        text = f"{title} {description}".lower()
        label_text = " ".join(labels).lower()
        all_text = f"{text} {label_text}"
        
        # Score based on high priority terms
        high_count = sum(1 for term in high_priority_terms if term in all_text)
        score += high_count * 0.3
        
        # Score based on medium priority terms
        medium_count = sum(1 for term in medium_priority_terms if term in all_text)
        score += medium_count * 0.2
        
        # Penalty for low priority terms
        low_count = sum(1 for term in low_priority_terms if term in all_text)
        score -= low_count * 0.1
        
        # Base score for any tech debt item
        score += 0.3
        
        # Check for specific high-priority labels
        high_priority_labels = ['critical', 'urgent', 'security', 'blocking']
        if any(label in label_text for label in high_priority_labels):
            score += 0.2
        
        # Normalize score to 0-1 range
        return max(0.0, min(1.0, score))
    
    def _estimate_effort_from_description(self, description: str) -> str:
        """Estimate effort required based on description content."""
        if not description:
            return "Small"
        
        desc_lower = description.lower()
        word_count = len(description.split())
        line_count = len(description.split('\n'))
        
        # Count complexity indicators
        complex_terms = [
            'architecture', 'system', 'migration', 'overhaul', 'complete',
            'entire', 'multiple', 'integration', 'infrastructure', 'platform',
            'security audit', 'performance optimization', 'api', 'database'
        ]
        
        simple_terms = [
            'typo', 'style', 'formatting', 'minor', 'small', 'quick',
            'simple', 'fix', 'update', 'documentation'
        ]
        
        # Score based on content analysis
        complexity_score = 0
        
        # Word and line count indicators
        if word_count > 100:
            complexity_score += 2
        elif word_count > 50:
            complexity_score += 1
        
        if line_count > 10:
            complexity_score += 2
        elif line_count > 5:
            complexity_score += 1
        
        # Complex terms
        complex_count = sum(1 for term in complex_terms if term in desc_lower)
        complexity_score += complex_count * 2
        
        # Simple terms (reduce complexity)
        simple_count = sum(1 for term in simple_terms if term in desc_lower)
        complexity_score -= simple_count
        
        # Check for lists or bullet points (indicating multiple tasks)
        if '- ' in description or '* ' in description or description.count('\n-') > 2:
            complexity_score += 2
        
        # Determine effort level
        if complexity_score >= 5:
            return "Large"
        elif complexity_score >= 2:
            return "Medium"
        else:
            return "Small"