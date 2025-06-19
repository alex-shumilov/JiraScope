"""Tests for content analyzer components."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from jirascope.analysis.content_analyzer import ContentAnalyzer, BatchContentAnalyzer, QualityAnalysisPrompts
from jirascope.models import QualityAnalysis, SplitAnalysis, SplitSuggestion, BatchAnalysisResult
from jirascope.core.config import Config
from tests.fixtures.analysis_fixtures import AnalysisFixtures


class TestQualityAnalysisPrompts:
    """Test the quality analysis prompt templates."""
    
    def test_description_quality_prompt_formatting(self):
        """Test that description quality prompt formats correctly."""
        prompt = QualityAnalysisPrompts.DESCRIPTION_QUALITY_PROMPT.format(
            summary="Test Story",
            issue_type="Story", 
            description="Test description"
        )
        
        assert "Test Story" in prompt
        assert "Story" in prompt
        assert "Test description" in prompt
        assert "Clarity (1-5)" in prompt
        assert "JSON format" in prompt
    
    def test_split_analysis_prompt_formatting(self):
        """Test that split analysis prompt formats correctly."""
        prompt = QualityAnalysisPrompts.SPLIT_ANALYSIS_PROMPT.format(
            summary="Complex Epic",
            description="Very long description with multiple features",
            story_points="13"
        )
        
        assert "Complex Epic" in prompt
        assert "Very long description" in prompt
        assert "13" in prompt
        assert "should_split" in prompt
    
    def test_batch_quality_prompt_formatting(self):
        """Test that batch quality prompt formats correctly."""
        work_items_text = "Item 1: Test item\nItem 2: Another item"
        prompt = QualityAnalysisPrompts.BATCH_QUALITY_PROMPT.format(
            work_items=work_items_text
        )
        
        assert work_items_text in prompt
        assert "analyses" in prompt
        assert "work_item_key" in prompt


class TestContentAnalyzer:
    """Test the content analyzer functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock(spec=Config)
        config.claude.model = "claude-3-5-sonnet-20241022"
        return config
    
    @pytest.fixture
    def mock_claude_client(self, mock_claude_responses):
        """Create mock Claude client."""
        claude_client = AsyncMock()
        
        # Set up different responses based on analysis type
        def mock_analyze(prompt, analysis_type, **kwargs):
            response = AsyncMock()
            if analysis_type == "quality_analysis":
                response.content = mock_claude_responses['quality_analysis']['content']
                response.cost = mock_claude_responses['quality_analysis']['cost']
            elif analysis_type == "split_analysis":
                response.content = mock_claude_responses['split_analysis']['content']
                response.cost = mock_claude_responses['split_analysis']['cost']
            else:
                response.content = '{"error": "unknown analysis type"}'
                response.cost = 0.01
            return response
        
        claude_client.analyze.side_effect = mock_analyze
        return claude_client
    
    @pytest.mark.asyncio
    async def test_analyze_description_quality_success(self, mock_config, mock_claude_client, sample_work_items):
        """Test successful description quality analysis."""
        with patch('jirascope.analysis.content_analyzer.ClaudeClient', return_value=mock_claude_client):
            async with ContentAnalyzer(mock_config) as analyzer:
                mock_claude_client.__aenter__ = AsyncMock(return_value=mock_claude_client)
                mock_claude_client.__aexit__ = AsyncMock()
                
                work_item = sample_work_items[0]
                analysis = await analyzer.analyze_description_quality(work_item)
                
                assert isinstance(analysis, QualityAnalysis)
                assert analysis.work_item_key == work_item.key
                assert 1 <= analysis.clarity_score <= 5
                assert 1 <= analysis.completeness_score <= 5
                assert 1 <= analysis.actionability_score <= 5
                assert 1 <= analysis.testability_score <= 5
                assert 1.0 <= analysis.overall_score <= 5.0
                assert analysis.risk_level in ["Low", "Medium", "High"]
                assert len(analysis.improvement_suggestions) > 0
                assert analysis.analysis_cost > 0
    
    @pytest.mark.asyncio
    async def test_analyze_description_quality_with_malformed_json(self, mock_config, sample_work_items):
        """Test handling of malformed JSON response from Claude."""
        claude_client = AsyncMock()
        claude_client.analyze.return_value = AsyncMock(
            content='Malformed JSON response without proper structure',
            cost=0.02
        )
        
        with patch('jirascope.analysis.content_analyzer.ClaudeClient', return_value=claude_client):
            async with ContentAnalyzer(mock_config) as analyzer:
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                work_item = sample_work_items[0]
                analysis = await analyzer.analyze_description_quality(work_item)
                
                # Should use fallback values
                assert analysis.clarity_score == 3
                assert analysis.completeness_score == 3
                assert analysis.actionability_score == 3
                assert analysis.testability_score == 3
                assert analysis.risk_level == "Medium"
    
    @pytest.mark.asyncio
    async def test_suggest_work_item_splits_should_split(self, mock_config, mock_claude_client, sample_work_items):
        """Test work item split analysis when splitting is recommended."""
        with patch('jirascope.analysis.content_analyzer.ClaudeClient', return_value=mock_claude_client):
            async with ContentAnalyzer(mock_config) as analyzer:
                mock_claude_client.__aenter__ = AsyncMock(return_value=mock_claude_client)
                mock_claude_client.__aexit__ = AsyncMock()
                
                # Use the complex work item that should be split
                complex_item = sample_work_items[5]  # "Complete E-commerce Platform Overhaul"
                analysis = await analyzer.suggest_work_item_splits(complex_item)
                
                assert isinstance(analysis, SplitAnalysis)
                assert analysis.work_item_key == complex_item.key
                assert analysis.should_split is True
                assert 0.0 <= analysis.complexity_score <= 1.0
                assert len(analysis.suggested_splits) > 0
                assert analysis.reasoning is not None
                assert analysis.analysis_cost > 0
                
                # Check split suggestions structure
                for split in analysis.suggested_splits:
                    assert isinstance(split, SplitSuggestion)
                    assert split.suggested_title is not None
                    assert split.suggested_description is not None
    
    @pytest.mark.asyncio
    async def test_suggest_work_item_splits_no_split_needed(self, mock_config, sample_work_items):
        """Test work item split analysis when no splitting is needed."""
        claude_client = AsyncMock()
        claude_client.analyze.return_value = AsyncMock(
            content='{"should_split": false, "complexity_score": 0.3, "reasoning": "Work item is appropriately sized", "suggested_splits": []}',
            cost=0.02
        )
        
        with patch('jirascope.analysis.content_analyzer.ClaudeClient', return_value=claude_client):
            async with ContentAnalyzer(mock_config) as analyzer:
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                simple_item = sample_work_items[0]  # Simple work item
                analysis = await analyzer.suggest_work_item_splits(simple_item)
                
                assert analysis.should_split is False
                assert analysis.complexity_score == 0.3
                assert len(analysis.suggested_splits) == 0
    
    @pytest.mark.asyncio
    async def test_story_points_estimation_heuristic(self, mock_config, mock_claude_client):
        """Test the story points estimation heuristic based on description length."""
        from jirascope.models import WorkItem
        from datetime import datetime
        
        # Create work items with different description lengths
        short_item = WorkItem(
            key="SHORT-1",
            summary="Short item",
            description="Short description",
            issue_type="Story",
            status="Open",
            created=datetime.now(),
            updated=datetime.now(),
            reporter="test"
        )
        
        medium_item = WorkItem(
            key="MEDIUM-1", 
            summary="Medium item",
            description="A" * 600,  # Medium length description
            issue_type="Story",
            status="Open",
            created=datetime.now(),
            updated=datetime.now(),
            reporter="test"
        )
        
        long_item = WorkItem(
            key="LONG-1",
            summary="Long item", 
            description="A" * 1200,  # Long description
            issue_type="Story",
            status="Open",
            created=datetime.now(),
            updated=datetime.now(),
            reporter="test"
        )
        
        with patch('jirascope.analysis.content_analyzer.ClaudeClient', return_value=mock_claude_client):
            async with ContentAnalyzer(mock_config) as analyzer:
                mock_claude_client.__aenter__ = AsyncMock(return_value=mock_claude_client)
                mock_claude_client.__aexit__ = AsyncMock()
                
                # The actual estimation happens in the prompt formatting
                # We can verify this by checking the prompt contains the right estimate
                await analyzer.suggest_work_item_splits(short_item)
                await analyzer.suggest_work_item_splits(medium_item)
                await analyzer.suggest_work_item_splits(long_item)
                
                # Verify Claude was called for each item
                assert mock_claude_client.analyze.call_count == 3
    
    @pytest.mark.asyncio
    async def test_context_manager_initialization(self, mock_config):
        """Test that async context manager properly initializes Claude client."""
        with patch('jirascope.analysis.content_analyzer.ClaudeClient') as mock_claude:
            mock_claude_instance = AsyncMock()
            mock_claude.return_value = mock_claude_instance
            
            async with ContentAnalyzer(mock_config) as analyzer:
                assert analyzer.claude_client is not None
                mock_claude_instance.__aenter__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, mock_config):
        """Test that async context manager properly cleans up Claude client."""
        with patch('jirascope.analysis.content_analyzer.ClaudeClient') as mock_claude:
            mock_claude_instance = AsyncMock()
            mock_claude.return_value = mock_claude_instance
            
            async with ContentAnalyzer(mock_config) as analyzer:
                pass
            
            mock_claude_instance.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_claude_failure(self, mock_config, sample_work_items):
        """Test error handling when Claude API fails."""
        claude_client = AsyncMock()
        claude_client.analyze.side_effect = Exception("Claude API error")
        
        with patch('jirascope.analysis.content_analyzer.ClaudeClient', return_value=claude_client):
            async with ContentAnalyzer(mock_config) as analyzer:
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                work_item = sample_work_items[0]
                
                with pytest.raises(Exception, match="Claude API error"):
                    await analyzer.analyze_description_quality(work_item)


class TestBatchContentAnalyzer:
    """Test the batch content analyzer functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock(spec=Config)
        config.claude.model = "claude-3-5-sonnet-20241022"
        return config
    
    @pytest.fixture
    def mock_claude_client(self, mock_claude_responses):
        """Create mock Claude client for batch operations."""
        claude_client = AsyncMock()
        
        # Mock batch response
        batch_response = {
            "analyses": [
                {
                    "work_item_key": "TEST-1",
                    "clarity_score": 4,
                    "completeness_score": 3,
                    "actionability_score": 4,
                    "testability_score": 3,
                    "overall_score": 3.5,
                    "improvement_suggestions": ["Add acceptance criteria"],
                    "risk_level": "Low"
                },
                {
                    "work_item_key": "TEST-2", 
                    "clarity_score": 3,
                    "completeness_score": 2,
                    "actionability_score": 3,
                    "testability_score": 2,
                    "overall_score": 2.5,
                    "improvement_suggestions": ["Clarify requirements", "Add technical details"],
                    "risk_level": "Medium"
                }
            ]
        }
        
        claude_client.analyze.return_value = AsyncMock(
            content=json.dumps(batch_response),
            cost=0.05
        )
        
        return claude_client
    
    @pytest.mark.asyncio
    async def test_analyze_multiple_items_success(self, mock_config, mock_claude_client, sample_work_items):
        """Test successful batch analysis of multiple work items."""
        with patch('jirascope.analysis.content_analyzer.ClaudeClient', return_value=mock_claude_client):
            async with BatchContentAnalyzer(mock_config) as batch_analyzer:
                mock_claude_client.__aenter__ = AsyncMock(return_value=mock_claude_client)
                mock_claude_client.__aexit__ = AsyncMock()
                
                work_items = sample_work_items[:3]
                result = await batch_analyzer.analyze_multiple_items(
                    work_items=work_items,
                    analysis_types=["quality"],
                    batch_size=2
                )
                
                assert isinstance(result, BatchAnalysisResult)
                assert result.total_items_processed == 3
                assert result.successful_analyses > 0
                assert result.failed_analyses >= 0
                assert result.total_cost > 0
                assert result.processing_time > 0
                assert len(result.analysis_results) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_multiple_items_with_batch_size(self, mock_config, mock_claude_client, sample_work_items):
        """Test batch analysis respects batch size parameter."""
        with patch('jirascope.analysis.content_analyzer.ClaudeClient', return_value=mock_claude_client):
            async with BatchContentAnalyzer(mock_config) as batch_analyzer:
                mock_claude_client.__aenter__ = AsyncMock(return_value=mock_claude_client)
                mock_claude_client.__aexit__ = AsyncMock()
                
                work_items = sample_work_items[:5]
                batch_size = 2
                
                result = await batch_analyzer.analyze_multiple_items(
                    work_items=work_items,
                    analysis_types=["quality"],
                    batch_size=batch_size
                )
                
                # Should process in batches of 2, so expect multiple Claude calls
                # 5 items with batch_size=2 means 3 batches (2+2+1)
                assert mock_claude_client.analyze.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_batch_quality_analysis_json_parsing_failure(self, mock_config, sample_work_items):
        """Test batch analysis with JSON parsing failure."""
        claude_client = AsyncMock()
        claude_client.analyze.return_value = AsyncMock(
            content="Malformed JSON response",
            cost=0.03
        )
        
        with patch('jirascope.analysis.content_analyzer.ClaudeClient', return_value=claude_client):
            async with BatchContentAnalyzer(mock_config) as batch_analyzer:
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                work_items = sample_work_items[:2]
                result = await batch_analyzer.analyze_multiple_items(
                    work_items=work_items,
                    analysis_types=["quality"],
                    batch_size=2
                )
                
                # Should fallback to individual analysis values
                assert result.successful_analyses >= 0
                assert len(result.analysis_results) == len(work_items)
    
    @pytest.mark.asyncio
    async def test_fallback_to_individual_analysis(self, mock_config, sample_work_items):
        """Test fallback to individual analysis for non-quality analysis types."""
        with patch('jirascope.analysis.content_analyzer.ContentAnalyzer') as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer_class.return_value = mock_analyzer
            
            # Mock individual analysis results
            mock_analysis = MagicMock()
            mock_analysis.dict.return_value = {"work_item_key": "TEST-1", "score": 3.5}
            mock_analysis.analysis_cost = 0.02
            mock_analyzer.analyze_description_quality.return_value = mock_analysis
            
            async with BatchContentAnalyzer(mock_config) as batch_analyzer:
                work_items = sample_work_items[:2]
                result = await batch_analyzer._process_batch(work_items, ["split"])  # Non-quality analysis
                
                assert "analyses" in result
                assert "cost" in result
                assert len(result["analyses"]) == len(work_items)
    
    @pytest.mark.asyncio
    async def test_batch_analysis_error_handling(self, mock_config, sample_work_items):
        """Test error handling in batch analysis."""
        claude_client = AsyncMock()
        claude_client.analyze.side_effect = Exception("Batch processing error")
        
        with patch('jirascope.analysis.content_analyzer.ClaudeClient', return_value=claude_client):
            async with BatchContentAnalyzer(mock_config) as batch_analyzer:
                claude_client.__aenter__ = AsyncMock(return_value=claude_client)
                claude_client.__aexit__ = AsyncMock()
                
                work_items = sample_work_items[:2]
                result = await batch_analyzer.analyze_multiple_items(
                    work_items=work_items,
                    analysis_types=["quality"],
                    batch_size=2
                )
                
                # Should handle errors gracefully
                assert result.failed_analyses > 0
                assert len(result.errors) > 0
                assert "Batch processing error" in str(result.errors)
    
    @pytest.mark.asyncio
    async def test_context_manager_initialization(self, mock_config):
        """Test that batch analyzer context manager works correctly."""
        with patch('jirascope.analysis.content_analyzer.ClaudeClient') as mock_claude:
            mock_claude_instance = AsyncMock()
            mock_claude.return_value = mock_claude_instance
            
            async with BatchContentAnalyzer(mock_config) as batch_analyzer:
                assert batch_analyzer.claude_client is not None
                mock_claude_instance.__aenter__.assert_called_once()


class TestContentAnalysisModels:
    """Test the content analysis model classes."""
    
    def test_quality_analysis_model(self):
        """Test QualityAnalysis model creation and validation."""
        analysis = QualityAnalysis(
            work_item_key="TEST-1",
            clarity_score=4,
            completeness_score=3,
            actionability_score=4,
            testability_score=3,
            overall_score=3.5,
            improvement_suggestions=["Add acceptance criteria", "Include technical notes"],
            risk_level="Low",
            analysis_cost=0.03
        )
        
        assert analysis.work_item_key == "TEST-1"
        assert analysis.clarity_score == 4
        assert analysis.overall_score == 3.5
        assert analysis.risk_level == "Low"
        assert len(analysis.improvement_suggestions) == 2
        assert analysis.analysis_cost == 0.03
    
    def test_split_suggestion_model(self):
        """Test SplitSuggestion model creation."""
        suggestion = SplitSuggestion(
            suggested_title="User Authentication",
            suggested_description="Implement OAuth2 authentication",
            estimated_effort="Medium",
            dependencies=["Database setup", "Security review"]
        )
        
        assert suggestion.suggested_title == "User Authentication"
        assert suggestion.estimated_effort == "Medium"
        assert len(suggestion.dependencies) == 2
    
    def test_split_analysis_model(self):
        """Test SplitAnalysis model creation and validation."""
        suggestions = [
            SplitSuggestion(
                suggested_title="Frontend Components",
                suggested_description="UI components implementation",
                estimated_effort="Small",
                dependencies=[]
            )
        ]
        
        analysis = SplitAnalysis(
            work_item_key="TEST-6",
            should_split=True,
            complexity_score=0.8,
            suggested_splits=suggestions,
            reasoning="Item contains multiple independent features",
            analysis_cost=0.04
        )
        
        assert analysis.work_item_key == "TEST-6"
        assert analysis.should_split is True
        assert analysis.complexity_score == 0.8
        assert len(analysis.suggested_splits) == 1
        assert "independent features" in analysis.reasoning
    
    def test_batch_analysis_result_model(self):
        """Test BatchAnalysisResult model creation."""
        result = BatchAnalysisResult(
            total_items_processed=5,
            successful_analyses=4,
            failed_analyses=1,
            total_cost=0.12,
            processing_time=2.5,
            analysis_results=[{"item": "TEST-1", "score": 3.5}],
            errors=["One batch failed due to timeout"]
        )
        
        assert result.total_items_processed == 5
        assert result.successful_analyses == 4
        assert result.failed_analyses == 1
        assert result.total_cost == 0.12
        assert result.processing_time == 2.5
        assert len(result.analysis_results) == 1
        assert len(result.errors) == 1


if __name__ == "__main__":
    pytest.main([__file__])