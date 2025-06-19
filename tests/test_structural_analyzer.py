"""Tests for structural analyzer components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from jirascope.analysis.structural_analyzer import StructuralAnalyzer, TechDebtClusterer
from jirascope.models import TechDebtReport, TechDebtCluster
from jirascope.core.config import Config
from tests.fixtures.analysis_fixtures import AnalysisFixtures


class TestTechDebtClusterer:
    """Test the tech debt clustering logic."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = Config(
            jira_mcp_endpoint="http://localhost:8080/mcp",
            lmstudio_endpoint="http://localhost:1234/v1", 
            qdrant_url="http://localhost:6333",
            claude_api_key="test-key"
        )
        self.clusterer = TechDebtClusterer(self.config)
        
    def test_clustering_parameters(self):
        """Test that clustering parameters are correctly set."""
        assert self.clusterer.min_samples >= 1
        assert self.clusterer.eps > 0
        assert self.clusterer.max_clusters > 0
    
    def test_identify_tech_debt_items(self, sample_work_items):
        """Test identification of tech debt work items."""
        tech_debt_items = self.clusterer._identify_tech_debt_items(sample_work_items)
        
        # Should identify items with tech debt indicators
        tech_debt_keys = {item.key for item in tech_debt_items}
        assert "TEST-3" in tech_debt_keys  # "Refactor legacy payment processing"
        assert "TEST-4" in tech_debt_keys  # "Cleanup outdated database queries"
        
        # Should exclude non-tech-debt items
        assert "TEST-5" not in tech_debt_keys  # High quality story
    
    def test_calculate_priority_score(self):
        """Test priority score calculation for tech debt items."""
        # High priority indicators
        high_priority_desc = "Critical legacy system causing daily outages and security vulnerabilities"
        high_score = self.clusterer._calculate_priority_score(
            "Critical system outage fix", high_priority_desc, ["critical", "security"]
        )
        assert high_score > 0.7
        
        # Low priority indicators  
        low_priority_desc = "Minor code style improvement"
        low_score = self.clusterer._calculate_priority_score(
            "Code style update", low_priority_desc, ["cleanup", "minor"]
        )
        assert low_score < 0.5
    
    def test_estimate_effort_from_description(self):
        """Test effort estimation based on description length and complexity."""
        # Simple task
        simple_desc = "Fix typo in documentation"
        simple_effort = self.clusterer._estimate_effort_from_description(simple_desc)
        assert simple_effort == "Small"
        
        # Medium complexity task
        medium_desc = "Refactor user authentication module to use new security library and update all dependent components"
        medium_effort = self.clusterer._estimate_effort_from_description(medium_desc)
        assert medium_effort == "Medium"
        
        # Complex task
        complex_desc = """Complete system architecture overhaul including:
        - Database migration to new schema
        - API endpoints restructuring  
        - Frontend component refactoring
        - Performance optimization
        - Security audit and fixes
        - Documentation updates
        - Testing suite implementation"""
        complex_effort = self.clusterer._estimate_effort_from_description(complex_desc)
        assert complex_effort == "Large"


class TestStructuralAnalyzer:
    """Test the structural analyzer functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock(spec=Config)
        config.claude_model = "claude-3-5-sonnet-20241022"
        return config
    
    @pytest.fixture
    def mock_clients(self, mock_claude_responses):
        """Create mock clients."""
        qdrant_client = AsyncMock()
        lm_client = AsyncMock()
        claude_client = AsyncMock()
        
        # Mock Qdrant search for tech debt items
        tech_debt_items = AnalysisFixtures.create_sample_work_items()[2:4]  # Tech debt items
        mock_points = []
        
        for item in tech_debt_items:
            point = MagicMock()
            point.payload = item.dict()
            point.payload['embedding'] = AnalysisFixtures.create_mock_embeddings()[0]
            mock_points.append(point)
        
        def mock_scroll(*args, **kwargs):
            return [mock_points], None
        
        qdrant_client.client.scroll.side_effect = mock_scroll
        
        # Mock embeddings generation
        mock_embeddings = AnalysisFixtures.create_mock_embeddings()
        lm_client.generate_embeddings.return_value = mock_embeddings[:4]
        lm_client.calculate_similarity.return_value = 0.75  # Similar items for clustering
        
        # Mock Claude analysis
        claude_client.analyze.return_value = AsyncMock(
            content=mock_claude_responses['tech_debt_cluster_analysis']['content'],
            cost=mock_claude_responses['tech_debt_cluster_analysis']['cost']
        )
        
        return qdrant_client, lm_client, claude_client
    
    @pytest.mark.asyncio
    async def test_identify_tech_debt_clusters_success(self, mock_config, mock_clients):
        """Test successful tech debt clustering analysis."""
        qdrant_client, lm_client, claude_client = mock_clients
        
        with patch('jirascope.analysis.structural_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.structural_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.structural_analyzer.ClaudeClient', return_value=claude_client):
            
            async with StructuralAnalyzer(mock_config) as analyzer:
                # Mock async context managers
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                report = await analyzer.identify_tech_debt_clusters()
                
                assert isinstance(report, TechDebtReport)
                assert report.total_tech_debt_items >= 0
                assert isinstance(report.clusters, list)
                assert report.analysis_cost > 0
    
    @pytest.mark.asyncio
    async def test_identify_tech_debt_clusters_with_project_filter(self, mock_config, mock_clients):
        """Test tech debt clustering with project filtering."""
        qdrant_client, lm_client, claude_client = mock_clients
        
        with patch('jirascope.analysis.structural_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.structural_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.structural_analyzer.ClaudeClient', return_value=claude_client):
            
            async with StructuralAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                report = await analyzer.identify_tech_debt_clusters(project_key="TEST")
                
                # Verify project filter was applied in Qdrant query
                calls = qdrant_client.client.scroll.call_args_list
                assert any("TEST" in str(call) for call in calls if call.kwargs.get('scroll_filter'))
    
    @pytest.mark.asyncio
    async def test_cluster_similar_tech_debt_items(self, mock_config, mock_clients, sample_work_items):
        """Test clustering of similar tech debt items."""
        qdrant_client, lm_client, claude_client = mock_clients
        
        # Use tech debt items from fixtures
        tech_debt_items = [item for item in sample_work_items if "refactor" in item.summary.lower() or "cleanup" in item.summary.lower()]
        
        with patch('jirascope.analysis.structural_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.structural_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.structural_analyzer.ClaudeClient', return_value=claude_client):
            
            async with StructuralAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                # Mock embeddings for clustering
                mock_embeddings = AnalysisFixtures.create_mock_embeddings()
                for i, item in enumerate(tech_debt_items):
                    item.embedding = mock_embeddings[i]
                
                clusters = await analyzer._cluster_similar_tech_debt_items(tech_debt_items)
                
                assert isinstance(clusters, list)
                for cluster in clusters:
                    assert isinstance(cluster, TechDebtCluster)
                    assert len(cluster.work_item_keys) > 0
                    assert cluster.theme is not None
                    assert 0.0 <= cluster.priority_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_analyze_cluster_with_claude(self, mock_config, mock_clients, sample_work_items):
        """Test Claude analysis of tech debt clusters."""
        qdrant_client, lm_client, claude_client = mock_clients
        
        tech_debt_items = sample_work_items[2:4]  # Tech debt items
        
        with patch('jirascope.analysis.structural_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.structural_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.structural_analyzer.ClaudeClient', return_value=claude_client):
            
            async with StructuralAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                analysis = await analyzer._analyze_cluster_with_claude(tech_debt_items)
                
                assert "theme" in analysis
                assert "priority_score" in analysis
                assert "estimated_effort" in analysis
                assert "dependencies" in analysis
                claude_client.analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_no_tech_debt_items_found(self, mock_config, mock_clients):
        """Test scenario where no tech debt items are found."""
        qdrant_client, lm_client, claude_client = mock_clients
        
        # Mock empty results from Qdrant
        def mock_empty_scroll(*args, **kwargs):
            return [[], None]
        
        qdrant_client.client.scroll.side_effect = mock_empty_scroll
        
        with patch('jirascope.analysis.structural_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.structural_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.structural_analyzer.ClaudeClient', return_value=claude_client):
            
            async with StructuralAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                report = await analyzer.identify_tech_debt_clusters()
                
                assert report.total_tech_debt_items == 0
                assert len(report.clusters) == 0
    
    @pytest.mark.asyncio
    async def test_context_manager_initialization(self, mock_config):
        """Test that async context manager properly initializes clients."""
        with patch('jirascope.analysis.structural_analyzer.QdrantVectorClient') as mock_qdrant, \
             patch('jirascope.analysis.structural_analyzer.LMStudioClient') as mock_lm, \
             patch('jirascope.analysis.structural_analyzer.ClaudeClient') as mock_claude:
            
            mock_qdrant_instance = AsyncMock()
            mock_lm_instance = AsyncMock()
            mock_claude_instance = AsyncMock()
            mock_qdrant.return_value = mock_qdrant_instance
            mock_lm.return_value = mock_lm_instance
            mock_claude.return_value = mock_claude_instance
            
            async with StructuralAnalyzer(mock_config) as analyzer:
                # Verify clients were created
                assert analyzer.qdrant_client is not None
                assert analyzer.lm_client is not None
                assert analyzer.claude_client is not None
                
                # Verify __aenter__ was called
                mock_qdrant_instance.__aenter__.assert_called_once()
                mock_lm_instance.__aenter__.assert_called_once()
                mock_claude_instance.__aenter__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, mock_config):
        """Test that async context manager properly cleans up clients."""
        with patch('jirascope.analysis.structural_analyzer.QdrantVectorClient') as mock_qdrant, \
             patch('jirascope.analysis.structural_analyzer.LMStudioClient') as mock_lm, \
             patch('jirascope.analysis.structural_analyzer.ClaudeClient') as mock_claude:
            
            mock_qdrant_instance = AsyncMock()
            mock_lm_instance = AsyncMock()
            mock_claude_instance = AsyncMock()
            mock_qdrant.return_value = mock_qdrant_instance
            mock_lm.return_value = mock_lm_instance
            mock_claude.return_value = mock_claude_instance
            
            async with StructuralAnalyzer(mock_config) as analyzer:
                pass
            
            # Verify __aexit__ was called
            mock_qdrant_instance.__aexit__.assert_called_once()
            mock_lm_instance.__aexit__.assert_called_once()
            mock_claude_instance.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_qdrant_failure(self, mock_config, mock_clients):
        """Test error handling when Qdrant operations fail."""
        qdrant_client, lm_client, claude_client = mock_clients
        
        # Mock Qdrant failure
        qdrant_client.client.scroll.side_effect = Exception("Qdrant connection error")
        
        with patch('jirascope.analysis.structural_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.structural_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.structural_analyzer.ClaudeClient', return_value=claude_client):
            
            async with StructuralAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                with pytest.raises(Exception, match="Qdrant connection error"):
                    await analyzer.identify_tech_debt_clusters()
    
    @pytest.mark.asyncio
    async def test_error_handling_claude_failure(self, mock_config, mock_clients, sample_work_items):
        """Test error handling when Claude analysis fails."""
        qdrant_client, lm_client, claude_client = mock_clients
        
        # Mock Claude failure
        claude_client.analyze.side_effect = Exception("Claude API error")
        
        tech_debt_items = sample_work_items[2:4]
        
        with patch('jirascope.analysis.structural_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.structural_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.structural_analyzer.ClaudeClient', return_value=claude_client):
            
            async with StructuralAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                # Should use fallback analysis when Claude fails
                analysis = await analyzer._analyze_cluster_with_claude(tech_debt_items)
                
                assert analysis["theme"] == "Tech Debt Cluster"
                assert analysis["priority_score"] == 0.5
    
    def test_tech_debt_cluster_model(self):
        """Test TechDebtCluster model creation and validation."""
        cluster = TechDebtCluster(
            cluster_id=1,
            theme="Legacy System Modernization",
            work_item_keys=["TEST-3", "TEST-4"],
            priority_score=0.75,
            estimated_effort="Large",
            dependencies=["database-migration", "api-updates"],
            impact_assessment="High impact on system performance",
            recommended_approach="Prioritize payment system refactor first"
        )
        
        assert cluster.cluster_id == 1
        assert cluster.theme == "Legacy System Modernization"
        assert len(cluster.work_item_keys) == 2
        assert "TEST-3" in cluster.work_item_keys
        assert cluster.priority_score == 0.75
        assert cluster.estimated_effort == "Large"
        assert "database-migration" in cluster.dependencies
        assert "performance" in cluster.impact_assessment
        assert "payment system" in cluster.recommended_approach
    
    def test_tech_debt_report_model(self):
        """Test TechDebtReport model creation and structure."""
        clusters = [
            TechDebtCluster(
                cluster_id=1,
                theme="Legacy System Updates",
                work_item_keys=["TEST-3", "TEST-4"],
                priority_score=0.8,
                estimated_effort="Medium",
                dependencies=[],
                impact_assessment="Medium impact",
                recommended_approach="Address incrementally"
            )
        ]
        
        report = TechDebtReport(
            total_tech_debt_items=5,
            clusters=clusters,
            processing_cost=0.15
        )
        
        assert report.total_tech_debt_items == 5
        assert len(report.clusters) == 1
        assert report.processing_cost == 0.15
    
    @pytest.mark.asyncio
    async def test_dbscan_clustering_algorithm(self, mock_config, mock_clients, sample_work_items):
        """Test DBSCAN clustering algorithm with mock embeddings."""
        qdrant_client, lm_client, claude_client = mock_clients
        
        # Create tech debt items with similar embeddings for clustering
        tech_debt_items = sample_work_items[2:5]  # 3 tech debt-related items
        mock_embeddings = AnalysisFixtures.create_mock_embeddings()
        
        # Make embeddings similar for clustering (first 3 items should cluster together)
        for i, item in enumerate(tech_debt_items):
            item.embedding = mock_embeddings[0]  # Same embedding for clustering
        
        with patch('jirascope.analysis.structural_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.structural_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.structural_analyzer.ClaudeClient', return_value=claude_client):
            
            async with StructuralAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                clusters = await analyzer._cluster_similar_tech_debt_items(tech_debt_items)
                
                # Should create at least one cluster
                assert len(clusters) >= 1
                
                # Check cluster properties
                for cluster in clusters:
                    assert len(cluster.work_item_keys) > 0
                    assert cluster.theme is not None
                    assert 0.0 <= cluster.priority_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])