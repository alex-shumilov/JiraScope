"""Tests for cross-epic analyzer components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from jirascope.analysis.cross_epic_analyzer import CrossEpicAnalyzer
from jirascope.models import CrossEpicReport, MisplacedWorkItem
from jirascope.core.config import Config
from tests.fixtures.analysis_fixtures import AnalysisFixtures


class TestCrossEpicAnalyzer:
    """Test the cross-epic analyzer functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock(spec=Config)
        config.lmstudio.embedding_model = "test-model"
        return config
    
    @pytest.fixture
    def mock_clients(self):
        """Create mock clients."""
        qdrant_client = AsyncMock()
        lm_client = AsyncMock()
        claude_client = AsyncMock()
        
        # Mock epic and work item data
        sample_hierarchies = AnalysisFixtures.create_sample_hierarchies()
        mock_embeddings = AnalysisFixtures.create_mock_embeddings()
        
        # Mock Qdrant scroll for epics and work items
        epic_points = []
        work_item_points = []
        
        for hierarchy in sample_hierarchies:
            # Epic points
            epic_point = MagicMock()
            epic_point.payload = hierarchy.epic.dict()
            epic_point.payload['embedding'] = mock_embeddings[0]  # Mock epic embedding
            epic_points.append(epic_point)
            
            # Work item points
            for item in hierarchy.all_items:
                work_item_point = MagicMock()
                work_item_point.payload = item.dict()
                work_item_point.payload['embedding'] = mock_embeddings[1]  # Mock work item embedding
                work_item_points.append(work_item_point)
        
        # Mock scroll responses
        def mock_scroll(*args, **kwargs):
            collection_name = kwargs.get('collection_name', '')
            if 'epic' in collection_name.lower():
                return [epic_points], None
            else:
                return [work_item_points], None
        
        qdrant_client.client.scroll.side_effect = mock_scroll
        
        # Mock LM Studio embeddings
        lm_client.generate_embeddings.return_value = mock_embeddings[:3]
        lm_client.calculate_similarity.return_value = 0.65  # Moderate similarity
        
        # Mock Claude analysis
        claude_client.analyze.return_value = AsyncMock(
            content='{"reasoning": "Work item seems better aligned with different epic based on content analysis", "confidence": 0.75}',
            cost=0.03
        )
        
        return qdrant_client, lm_client, claude_client
    
    @pytest.mark.asyncio
    async def test_find_misplaced_work_items_success(self, mock_config, mock_clients):
        """Test successful cross-epic analysis."""
        qdrant_client, lm_client, claude_client = mock_clients
        
        with patch('jirascope.analysis.cross_epic_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.cross_epic_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.cross_epic_analyzer.ClaudeClient', return_value=claude_client):
            
            async with CrossEpicAnalyzer(mock_config) as analyzer:
                # Mock async context managers
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                report = await analyzer.find_misplaced_work_items()
                
                assert isinstance(report, CrossEpicReport)
                assert report.epics_analyzed >= 0
                assert isinstance(report.misplaced_items, list)
    
    @pytest.mark.asyncio
    async def test_find_misplaced_work_items_with_project_filter(self, mock_config, mock_clients):
        """Test cross-epic analysis with project filtering."""
        qdrant_client, lm_client, claude_client = mock_clients
        
        with patch('jirascope.analysis.cross_epic_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.cross_epic_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.cross_epic_analyzer.ClaudeClient', return_value=claude_client):
            
            async with CrossEpicAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                report = await analyzer.find_misplaced_work_items(project_key="TEST")
                
                # Verify project filter was applied in Qdrant queries
                calls = qdrant_client.client.scroll.call_args_list
                assert any("TEST" in str(call) for call in calls if call.kwargs.get('scroll_filter'))
    
    @pytest.mark.asyncio
    async def test_calculate_epic_theme_embedding(self, mock_config, mock_clients):
        """Test epic theme embedding calculation."""
        qdrant_client, lm_client, claude_client = mock_clients
        sample_epics = AnalysisFixtures.create_sample_epics()
        sample_work_items = AnalysisFixtures.create_sample_work_items()[:3]
        
        with patch('jirascope.analysis.cross_epic_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.cross_epic_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.cross_epic_analyzer.ClaudeClient', return_value=claude_client):
            
            async with CrossEpicAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                theme_embedding = await analyzer._calculate_epic_theme_embedding(
                    sample_epics[0], sample_work_items
                )
                
                assert theme_embedding is not None
                assert len(theme_embedding) > 0  # Should return an embedding vector
                lm_client.generate_embeddings.assert_called()
    
    @pytest.mark.asyncio
    async def test_calculate_epic_coherence_score(self, mock_config, mock_clients):
        """Test epic coherence score calculation."""
        qdrant_client, lm_client, claude_client = mock_clients
        sample_work_items = AnalysisFixtures.create_sample_work_items()[:3]
        mock_embeddings = AnalysisFixtures.create_mock_embeddings()
        
        # Mock embeddings for work items
        for i, item in enumerate(sample_work_items):
            item.embedding = mock_embeddings[i]
        
        with patch('jirascope.analysis.cross_epic_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.cross_epic_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.cross_epic_analyzer.ClaudeClient', return_value=claude_client):
            
            async with CrossEpicAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                epic_theme = mock_embeddings[0]
                coherence_score = await analyzer._calculate_epic_coherence_score(
                    epic_theme, sample_work_items
                )
                
                assert 0.0 <= coherence_score <= 1.0
                lm_client.calculate_similarity.assert_called()
    
    @pytest.mark.asyncio
    async def test_analyze_misplacement_with_claude(self, mock_config, mock_clients):
        """Test Claude analysis for work item misplacement."""
        qdrant_client, lm_client, claude_client = mock_clients
        sample_work_items = AnalysisFixtures.create_sample_work_items()
        sample_epics = AnalysisFixtures.create_sample_epics()
        
        with patch('jirascope.analysis.cross_epic_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.cross_epic_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.cross_epic_analyzer.ClaudeClient', return_value=claude_client):
            
            async with CrossEpicAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                analysis = await analyzer._analyze_misplacement_with_claude(
                    sample_work_items[0], sample_epics[0], sample_epics[1]
                )
                
                assert "reasoning" in analysis
                assert "confidence" in analysis
                claude_client.analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager_initialization(self, mock_config):
        """Test that async context manager properly initializes clients."""
        with patch('jirascope.analysis.cross_epic_analyzer.QdrantVectorClient') as mock_qdrant, \
             patch('jirascope.analysis.cross_epic_analyzer.LMStudioClient') as mock_lm, \
             patch('jirascope.analysis.cross_epic_analyzer.ClaudeClient') as mock_claude:
            
            mock_qdrant_instance = AsyncMock()
            mock_lm_instance = AsyncMock()
            mock_claude_instance = AsyncMock()
            mock_qdrant.return_value = mock_qdrant_instance
            mock_lm.return_value = mock_lm_instance
            mock_claude.return_value = mock_claude_instance
            
            async with CrossEpicAnalyzer(mock_config) as analyzer:
                # Verify clients were created and initialized
                assert analyzer.qdrant_client is not None
                assert analyzer.lm_client is not None
                assert analyzer.claude_client is not None
                
                # Verify __aenter__ was called on all clients
                mock_qdrant_instance.__aenter__.assert_called_once()
                mock_lm_instance.__aenter__.assert_called_once()
                mock_claude_instance.__aenter__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, mock_config):
        """Test that async context manager properly cleans up clients."""
        with patch('jirascope.analysis.cross_epic_analyzer.QdrantVectorClient') as mock_qdrant, \
             patch('jirascope.analysis.cross_epic_analyzer.LMStudioClient') as mock_lm, \
             patch('jirascope.analysis.cross_epic_analyzer.ClaudeClient') as mock_claude:
            
            mock_qdrant_instance = AsyncMock()
            mock_lm_instance = AsyncMock()
            mock_claude_instance = AsyncMock()
            mock_qdrant.return_value = mock_qdrant_instance
            mock_lm.return_value = mock_lm_instance
            mock_claude.return_value = mock_claude_instance
            
            async with CrossEpicAnalyzer(mock_config) as analyzer:
                pass
            
            # Verify __aexit__ was called on all clients
            mock_qdrant_instance.__aexit__.assert_called_once()
            mock_lm_instance.__aexit__.assert_called_once()
            mock_claude_instance.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_qdrant_failure(self, mock_config, mock_clients):
        """Test error handling when Qdrant operations fail."""
        qdrant_client, lm_client, claude_client = mock_clients
        
        # Mock Qdrant failure
        qdrant_client.client.scroll.side_effect = Exception("Qdrant connection error")
        
        with patch('jirascope.analysis.cross_epic_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.cross_epic_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.cross_epic_analyzer.ClaudeClient', return_value=claude_client):
            
            async with CrossEpicAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                with pytest.raises(Exception, match="Qdrant connection error"):
                    await analyzer.find_misplaced_work_items()
    
    @pytest.mark.asyncio
    async def test_error_handling_claude_failure(self, mock_config, mock_clients):
        """Test error handling when Claude analysis fails."""
        qdrant_client, lm_client, claude_client = mock_clients
        sample_work_items = AnalysisFixtures.create_sample_work_items()
        sample_epics = AnalysisFixtures.create_sample_epics()
        
        # Mock Claude failure
        claude_client.analyze.side_effect = Exception("Claude API error")
        
        with patch('jirascope.analysis.cross_epic_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.cross_epic_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.cross_epic_analyzer.ClaudeClient', return_value=claude_client):
            
            async with CrossEpicAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                # Should return fallback analysis when Claude fails
                analysis = await analyzer._analyze_misplacement_with_claude(
                    sample_work_items[0], sample_epics[0], sample_epics[1]
                )
                
                assert "reasoning" in analysis
                assert "Unable to analyze" in analysis["reasoning"]
    
    def test_misplaced_work_item_model(self):
        """Test MisplacedWorkItem model creation and validation."""
        misplaced_item = MisplacedWorkItem(
            work_item_key="TEST-1",
            current_epic_key="EPIC-A",
            suggested_epic_key="EPIC-B",
            confidence_score=0.75,
            reasoning="Better thematic alignment with Epic B based on content analysis"
        )
        
        assert misplaced_item.work_item_key == "TEST-1"
        assert misplaced_item.current_epic_key == "EPIC-A"
        assert misplaced_item.suggested_epic_key == "EPIC-B"
        assert misplaced_item.confidence_score == 0.75
        assert "thematic alignment" in misplaced_item.reasoning
    
    def test_cross_epic_report_model(self):
        """Test CrossEpicReport model creation and structure."""
        misplaced_items = [
            MisplacedWorkItem(
                work_item_key="TEST-1",
                current_epic_key="EPIC-A",
                suggested_epic_key="EPIC-B",
                confidence_score=0.75,
                reasoning="Better alignment with Epic B"
            )
        ]
        
        report = CrossEpicReport(
            epics_analyzed=3,
            misplaced_items=misplaced_items,
            processing_cost=0.08
        )
        
        assert report.epics_analyzed == 3
        assert len(report.misplaced_items) == 1
        assert report.processing_cost == 0.08
    
    @pytest.mark.asyncio
    async def test_no_misplaced_items_found(self, mock_config, mock_clients):
        """Test scenario where no misplaced items are found."""
        qdrant_client, lm_client, claude_client = mock_clients
        
        # Mock high coherence scores (no misplacement)
        lm_client.calculate_similarity.return_value = 0.95
        
        with patch('jirascope.analysis.cross_epic_analyzer.QdrantVectorClient', return_value=qdrant_client), \
             patch('jirascope.analysis.cross_epic_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.cross_epic_analyzer.ClaudeClient', return_value=claude_client):
            
            async with CrossEpicAnalyzer(mock_config) as analyzer:
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                report = await analyzer.find_misplaced_work_items()
                
                assert len(report.misplaced_items) == 0
                assert report.epics_analyzed >= 0


if __name__ == "__main__":
    pytest.main([__file__])