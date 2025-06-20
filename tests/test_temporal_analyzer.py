"""Tests for temporal analyzer components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from jirascope.analysis.temporal_analyzer import TemporalAnalyzer, ScopeDriftDetector
from jirascope.models import ScopeDriftAnalysis, ScopeDriftEvent
from jirascope.core.config import Config
from tests.fixtures.analysis_fixtures import AnalysisFixtures


class TestScopeDriftDetector:
    """Test the scope drift detection logic."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = Config(
            jira_mcp_endpoint="http://localhost:8080/mcp",
            lmstudio_endpoint="http://localhost:1234/v1", 
            qdrant_url="http://localhost:6333",
            claude_api_key="test-key"
        )
        self.detector = ScopeDriftDetector(self.config)
        
    @pytest.mark.asyncio
    async def test_calculate_semantic_similarity(self):
        """Test semantic similarity calculation."""
        original_text = "Simple user login form"
        current_text = "Complete authentication system with OAuth2, 2FA, password reset, social logins, and security auditing"
        
        async with self.detector:
            # Mock the LM client's calculate_similarity method
            self.detector.lm_client.calculate_similarity = AsyncMock(return_value=0.25)  # Low similarity
            
            similarity = await self.detector._calculate_semantic_similarity(
                original_text, current_text
            )
            
            assert 0.0 <= similarity <= 1.0
            assert similarity == 0.25  # Should return the mocked value
    
    @pytest.mark.asyncio
    async def test_calculate_overall_drift(self):
        """Test overall drift calculation from multiple events."""
        from jirascope.models import ScopeDriftEvent
        
        # Create sample drift events
        drift_events = [
            ScopeDriftEvent(
                timestamp=datetime.now(),
                change_type="description_change",
                old_value="Simple login",
                new_value="Complex authentication system",
                drift_score=0.8,
                change_significance="major"
            ),
            ScopeDriftEvent(
                timestamp=datetime.now(),
                change_type="scope_expansion",
                old_value="Basic form",
                new_value="OAuth integration",
                drift_score=0.6,
                change_significance="moderate"
            )
        ]
        
        overall_drift = self.detector._calculate_overall_drift(drift_events)
        
        assert 0.0 <= overall_drift <= 1.0
        assert overall_drift > 0.5  # Should reflect significant drift


class TestTemporalAnalyzer:
    """Test the temporal analyzer functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock(spec=Config)
        config.claude_model = "claude-3-5-sonnet-20241022"
        return config
    
    @pytest.fixture
    def mock_clients(self, mock_claude_responses):
        """Create mock clients."""
        jira_client = AsyncMock()
        lm_client = AsyncMock()
        claude_client = AsyncMock()
        
        # Mock change history
        scope_drift_history = AnalysisFixtures.create_scope_drift_history()
        jira_client.get_work_item_change_history.return_value = scope_drift_history
        
        # Mock embedding generation
        mock_embeddings = AnalysisFixtures.create_mock_embeddings()
        lm_client.generate_embeddings.return_value = mock_embeddings[:3]
        lm_client.calculate_similarity.return_value = 0.25  # Low similarity for drift
        
        # Mock Claude analysis
        claude_client.analyze.return_value = AsyncMock(
            content=mock_claude_responses['scope_change_analysis']['content'],
            cost=mock_claude_responses['scope_change_analysis']['cost']
        )
        
        return jira_client, lm_client, claude_client
    
    @pytest.mark.asyncio
    async def test_analyze_scope_drift_success(self, mock_config, mock_clients, sample_work_items):
        """Test successful scope drift analysis."""
        jira_client, lm_client, claude_client = mock_clients
        
        with patch('jirascope.analysis.temporal_analyzer.JiraClient', return_value=jira_client), \
             patch('jirascope.analysis.temporal_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.temporal_analyzer.ClaudeClient', return_value=claude_client):
            
            async with TemporalAnalyzer(mock_config) as analyzer:
                # Mock async context managers
                jira_client.__aenter__ = AsyncMock(return_value=jira_client)
                jira_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                work_item = sample_work_items[0]
                analysis = await analyzer.analyze_scope_drift(work_item)
                
                assert isinstance(analysis, DriftAnalysis)
                assert analysis.work_item_key == work_item.key
                assert 0.0 <= analysis.drift_score <= 1.0
                assert analysis.drift_level in ["none", "minor", "moderate", "major", "critical"]
                assert len(analysis.change_events) > 0
                assert analysis.analysis_cost > 0
    
    @pytest.mark.asyncio
    async def test_analyze_scope_drift_no_history(self, mock_config, mock_clients, sample_work_items):
        """Test scope drift analysis with no change history."""
        jira_client, lm_client, claude_client = mock_clients
        
        # Mock empty change history
        jira_client.get_work_item_change_history.return_value = []
        
        with patch('jirascope.analysis.temporal_analyzer.JiraClient', return_value=jira_client), \
             patch('jirascope.analysis.temporal_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.temporal_analyzer.ClaudeClient', return_value=claude_client):
            
            async with TemporalAnalyzer(mock_config) as analyzer:
                jira_client.__aenter__ = AsyncMock(return_value=jira_client)
                jira_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                work_item = sample_work_items[0]
                analysis = await analyzer.analyze_scope_drift(work_item)
                
                assert analysis.drift_score == 0.0
                assert analysis.drift_level == "none"
                assert len(analysis.change_events) == 0
    
    @pytest.mark.asyncio
    async def test_detect_scope_drift_for_project_success(self, mock_config, mock_clients, sample_work_items):
        """Test scope drift detection for entire project."""
        jira_client, lm_client, claude_client = mock_clients
        
        # Mock work items for project
        jira_client.get_project_work_items.return_value = sample_work_items[:3]
        
        with patch('jirascope.analysis.temporal_analyzer.JiraClient', return_value=jira_client), \
             patch('jirascope.analysis.temporal_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.temporal_analyzer.ClaudeClient', return_value=claude_client):
            
            async with TemporalAnalyzer(mock_config) as analyzer:
                jira_client.__aenter__ = AsyncMock(return_value=jira_client)
                jira_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                report = await analyzer.detect_scope_drift_for_project("TEST")
                
                assert isinstance(report, ScopeDriftReport)
                assert report.project_key == "TEST"
                assert report.total_items_analyzed == 3
                assert report.items_with_drift >= 0
                assert len(report.drift_analyses) == 3
                assert report.total_analysis_cost > 0
    
    @pytest.mark.asyncio
    async def test_detect_scope_drift_with_time_range(self, mock_config, mock_clients, sample_work_items):
        """Test scope drift detection with time range filtering."""
        jira_client, lm_client, claude_client = mock_clients
        
        # Mock filtered work items
        jira_client.get_project_work_items.return_value = sample_work_items[:2]
        
        with patch('jirascope.analysis.temporal_analyzer.JiraClient', return_value=jira_client), \
             patch('jirascope.analysis.temporal_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.temporal_analyzer.ClaudeClient', return_value=claude_client):
            
            async with TemporalAnalyzer(mock_config) as analyzer:
                jira_client.__aenter__ = AsyncMock(return_value=jira_client)
                jira_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                start_date = datetime(2024, 1, 1)
                end_date = datetime(2024, 1, 31)
                
                report = await analyzer.detect_scope_drift_for_project(
                    "TEST", start_date=start_date, end_date=end_date
                )
                
                # Verify time range was passed to Jira client
                jira_client.get_project_work_items.assert_called_with(
                    "TEST", start_date=start_date, end_date=end_date
                )
                assert report.total_items_analyzed == 2
    
    def test_temporal_analyzer_initialization(self, mock_config):
        """Test that TemporalAnalyzer properly initializes."""
        analyzer = TemporalAnalyzer(mock_config)
        
        # Verify components were created
        assert analyzer.drift_detector is not None
        assert analyzer.config is not None
    
    @pytest.mark.asyncio
    async def test_epic_evolution_analysis(self, mock_config):
        """Test epic evolution analysis."""
        from jirascope.models import EvolutionReport
        
        analyzer = TemporalAnalyzer(mock_config)
        
        # Test the mock implementation
        report = await analyzer.epic_evolution_analysis("EPIC-123", days=30)
        
        assert isinstance(report, EvolutionReport)
        assert report.epic_key == "EPIC-123"
        assert report.time_period_days == 30
        assert isinstance(report.coherence_trend, list)
        assert isinstance(report.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_error_handling_jira_failure(self, mock_config, mock_clients, sample_work_items):
        """Test error handling when Jira API fails."""
        jira_client, lm_client, claude_client = mock_clients
        
        # Mock Jira failure
        jira_client.get_work_item_change_history.side_effect = Exception("Jira API error")
        
        with patch('jirascope.analysis.temporal_analyzer.JiraClient', return_value=jira_client), \
             patch('jirascope.analysis.temporal_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.temporal_analyzer.ClaudeClient', return_value=claude_client):
            
            async with TemporalAnalyzer(mock_config) as analyzer:
                jira_client.__aenter__ = AsyncMock(return_value=jira_client)
                jira_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                work_item = sample_work_items[0]
                
                with pytest.raises(Exception, match="Jira API error"):
                    await analyzer.analyze_scope_drift(work_item)
    
    @pytest.mark.asyncio
    async def test_error_handling_claude_failure(self, mock_config, mock_clients, sample_work_items):
        """Test error handling when Claude analysis fails."""
        jira_client, lm_client, claude_client = mock_clients
        
        # Mock Claude failure
        claude_client.analyze.side_effect = Exception("Claude API error")
        
        with patch('jirascope.analysis.temporal_analyzer.JiraClient', return_value=jira_client), \
             patch('jirascope.analysis.temporal_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.temporal_analyzer.ClaudeClient', return_value=claude_client):
            
            async with TemporalAnalyzer(mock_config) as analyzer:
                jira_client.__aenter__ = AsyncMock(return_value=jira_client)
                jira_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                work_item = sample_work_items[0]
                analysis = await analyzer.analyze_scope_drift(work_item)
                
                # Should use fallback analysis when Claude fails
                assert analysis.claude_insights == "Analysis unavailable due to API error"
    
    def test_change_event_model(self):
        """Test ChangeEvent model creation and validation."""
        change_event = ChangeEvent(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            field_changed="description",
            old_value="Simple login form",
            new_value="Login form with OAuth2 support",
            author="developer"
        )
        
        assert change_event.timestamp == datetime(2024, 1, 15, 10, 30, 0)
        assert change_event.field_changed == "description"
        assert "Simple login form" in change_event.old_value
        assert "OAuth2" in change_event.new_value
        assert change_event.author == "developer"
    
    def test_drift_analysis_model(self):
        """Test DriftAnalysis model creation and validation."""
        change_events = [
            ChangeEvent(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                field_changed="description",
                old_value="Simple form",
                new_value="Complex system",
                author="developer"
            )
        ]
        
        analysis = DriftAnalysis(
            work_item_key="TEST-1",
            drift_score=0.75,
            drift_level="major",
            change_events=change_events,
            semantic_similarity=0.25,
            complexity_increase=0.80,
            claude_insights="Significant scope expansion detected",
            analysis_cost=0.025
        )
        
        assert analysis.work_item_key == "TEST-1"
        assert analysis.drift_score == 0.75
        assert analysis.drift_level == "major"
        assert len(analysis.change_events) == 1
        assert analysis.semantic_similarity == 0.25
        assert analysis.complexity_increase == 0.80
        assert "expansion" in analysis.claude_insights
        assert analysis.analysis_cost == 0.025
    
    def test_scope_drift_report_model(self):
        """Test ScopeDriftReport model creation and structure."""
        drift_analyses = [
            DriftAnalysis(
                work_item_key="TEST-1",
                drift_score=0.75,
                drift_level="major",
                change_events=[],
                semantic_similarity=0.25,
                complexity_increase=0.80,
                claude_insights="Scope expansion",
                analysis_cost=0.025
            )
        ]
        
        report = ScopeDriftReport(
            project_key="TEST",
            total_items_analyzed=5,
            items_with_drift=1,
            drift_analyses=drift_analyses,
            total_analysis_cost=0.125
        )
        
        assert report.project_key == "TEST"
        assert report.total_items_analyzed == 5
        assert report.items_with_drift == 1
        assert len(report.drift_analyses) == 1
        assert report.total_analysis_cost == 0.125
    
    @pytest.mark.asyncio
    async def test_batch_analysis_performance(self, mock_config, mock_clients, sample_work_items):
        """Test performance with batch analysis of multiple items."""
        jira_client, lm_client, claude_client = mock_clients
        
        # Mock larger dataset
        large_work_items = sample_work_items * 5  # 40 items total
        jira_client.get_project_work_items.return_value = large_work_items
        
        with patch('jirascope.analysis.temporal_analyzer.JiraClient', return_value=jira_client), \
             patch('jirascope.analysis.temporal_analyzer.LMStudioClient', return_value=lm_client), \
             patch('jirascope.analysis.temporal_analyzer.ClaudeClient', return_value=claude_client):
            
            async with TemporalAnalyzer(mock_config) as analyzer:
                jira_client.__aenter__ = AsyncMock(return_value=jira_client)
                jira_client.__aexit__ = AsyncMock()
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                report = await analyzer.detect_scope_drift_for_project("TEST")
                
                assert report.total_items_analyzed == 40
                assert len(report.drift_analyses) == 40
                # Verify reasonable performance characteristics
                assert report.total_analysis_cost > 0


if __name__ == "__main__":
    pytest.main([__file__])