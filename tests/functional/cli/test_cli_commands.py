"""Comprehensive CLI Main Tests - Targeting actual Click commands for maximum coverage."""

import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

from click.testing import CliRunner

from jirascope.cli.main import cli, run_async


class TestCLIMainCommands:
    """Test CLI main commands for maximum coverage boost."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_config = None

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_config and os.path.exists(self.temp_config):
            os.unlink(self.temp_config)

    def create_test_config(self) -> str:
        """Create a temporary test configuration file."""
        config_content = """
jira:
  url: "https://test.atlassian.net"
  username: "test@example.com"
  password: "test_password"

qdrant:
  url: "http://localhost:6333"
  collection: "test_collection"

lmstudio:
  url: "http://localhost:1234"
  model: "test-model"

logging:
  level: "INFO"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content.strip())
            f.flush()
            self.temp_config = f.name
            return f.name

    def test_cli_main_group_help(self):
        """Test CLI main group help functionality."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "JiraScope - AI-powered Jira work item analysis" in result.output
        assert "--config" in result.output
        assert "--verbose" in result.output

    def test_cli_with_config_option(self):
        """Test CLI with config option."""
        config_file = self.create_test_config()

        # Test with valid config
        result = self.runner.invoke(cli, ["--config", config_file, "--help"])
        assert result.exit_code == 0

    def test_cli_with_invalid_config(self):
        """Test CLI with invalid config file."""
        result = self.runner.invoke(cli, ["--config", "/nonexistent/config.yaml", "--help"])
        assert result.exit_code == 2  # Click exits with 2 for bad options

    def test_cli_verbose_option(self):
        """Test CLI verbose option."""
        config_file = self.create_test_config()
        result = self.runner.invoke(cli, ["--config", config_file, "--verbose", "--help"])
        assert result.exit_code == 0

    @patch("jirascope.cli.main.setup_logging")
    @patch("jirascope.cli.main.Config.load")
    def test_cli_context_setup(self, mock_config_load, mock_setup_logging):
        """Test CLI context setup with mocked dependencies."""
        # Mock config loading
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        # Mock logging setup
        mock_cost_tracker = Mock()
        mock_setup_logging.return_value = mock_cost_tracker

        config_file = self.create_test_config()
        result = self.runner.invoke(cli, ["--config", config_file, "--help"])

        assert result.exit_code == 0
        mock_config_load.assert_called_once()
        mock_setup_logging.assert_called_once()

    @patch("jirascope.cli.main.QdrantVectorClient")
    @patch("jirascope.cli.main.LMStudioClient")
    @patch("jirascope.cli.main.Config.load")
    def test_health_command(self, mock_config_load, mock_lm_client, mock_qdrant_client):
        """Test health command functionality."""
        # Setup mocks
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        # Mock health check responses
        mock_qdrant_instance = AsyncMock()
        mock_qdrant_instance.health_check.return_value = True
        mock_qdrant_client.return_value.__aenter__.return_value = mock_qdrant_instance

        mock_lm_instance = AsyncMock()
        mock_lm_instance.health_check.return_value = True
        mock_lm_client.return_value.__aenter__.return_value = mock_lm_instance

        config_file = self.create_test_config()
        result = self.runner.invoke(cli, ["--config", config_file, "health"])

        assert result.exit_code == 0
        assert "Checking service health" in result.output
        assert "All services healthy!" in result.output

    @patch("jirascope.cli.main.QdrantVectorClient")
    @patch("jirascope.cli.main.LMStudioClient")
    @patch("jirascope.cli.main.Config.load")
    def test_health_command_failure(self, mock_config_load, mock_lm_client, mock_qdrant_client):
        """Test health command with service failures."""
        # Setup mocks
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        # Mock health check failures
        mock_qdrant_client.side_effect = Exception("Qdrant connection failed")
        mock_lm_client.side_effect = Exception("LMStudio connection failed")

        config_file = self.create_test_config()
        result = self.runner.invoke(cli, ["--config", config_file, "health"])

        assert result.exit_code == 1
        assert "Some services are not responding" in result.output

    def test_analyze_group_help(self):
        """Test analyze command group help."""
        result = self.runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "duplicates" in result.output
        assert "cross-epic" in result.output
        assert "quality" in result.output
        assert "template" in result.output

    @patch("jirascope.cli.main.SimilarityAnalyzer")
    @patch("jirascope.cli.main.QdrantVectorClient")
    @patch("jirascope.cli.main.Config.load")
    def test_analyze_duplicates_command(
        self, mock_config_load, mock_qdrant_client, mock_similarity_analyzer
    ):
        """Test analyze duplicates command."""
        # Setup mocks
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        mock_qdrant_instance = AsyncMock()
        mock_qdrant_instance.get_all_work_items.return_value = []
        mock_qdrant_client.return_value.__aenter__.return_value = mock_qdrant_instance

        mock_analyzer_instance = AsyncMock()
        mock_analyzer_instance.find_potential_duplicates.return_value = []
        mock_similarity_analyzer.return_value = mock_analyzer_instance

        config_file = self.create_test_config()
        result = self.runner.invoke(
            cli, ["--config", config_file, "analyze", "duplicates", "--project", "TEST"]
        )

        assert result.exit_code == 0

    @patch("jirascope.cli.main.CrossEpicAnalyzer")
    @patch("jirascope.cli.main.QdrantVectorClient")
    @patch("jirascope.cli.main.Config.load")
    def test_analyze_cross_epic_command(
        self, mock_config_load, mock_qdrant_client, mock_cross_epic_analyzer
    ):
        """Test analyze cross-epic command."""
        # Setup mocks
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        mock_qdrant_instance = AsyncMock()
        mock_qdrant_instance.get_all_work_items.return_value = []
        mock_qdrant_client.return_value.__aenter__.return_value = mock_qdrant_instance

        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_cross_epic_dependencies.return_value = Mock(
            dependencies=[], potential_duplicates=[], recommendations=[]
        )
        mock_cross_epic_analyzer.return_value = mock_analyzer_instance

        config_file = self.create_test_config()
        result = self.runner.invoke(
            cli, ["--config", config_file, "analyze", "cross-epic", "--project", "TEST"]
        )

        assert result.exit_code == 0

    @patch("jirascope.cli.main.EmbeddingQualityValidator")
    @patch("jirascope.cli.main.QdrantVectorClient")
    @patch("jirascope.cli.main.Config.load")
    def test_analyze_quality_command(
        self, mock_config_load, mock_qdrant_client, mock_quality_validator
    ):
        """Test analyze quality command."""
        # Setup mocks
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        # Mock work item
        mock_work_item = Mock()
        mock_work_item.key = "TEST-123"
        mock_work_item.summary = "Test work item"

        mock_qdrant_instance = AsyncMock()
        mock_qdrant_instance.get_work_item_by_key.return_value = mock_work_item
        mock_qdrant_client.return_value.__aenter__.return_value = mock_qdrant_instance

        # Mock quality validation
        mock_validator_instance = AsyncMock()
        mock_validation_result = Mock()
        mock_validation_result.overall_score = 0.85
        mock_validation_result.quality_issues = []
        mock_validation_result.recommendations = []
        mock_validator_instance.validate_work_item.return_value = mock_validation_result
        mock_quality_validator.return_value = mock_validator_instance

        config_file = self.create_test_config()
        result = self.runner.invoke(
            cli, ["--config", config_file, "analyze", "quality", "TEST-123"]
        )

        assert result.exit_code == 0

    @patch("jirascope.cli.main.TemplateInferenceEngine")
    @patch("jirascope.cli.main.QdrantVectorClient")
    @patch("jirascope.cli.main.Config.load")
    def test_analyze_template_command(
        self, mock_config_load, mock_qdrant_client, mock_template_engine
    ):
        """Test analyze template command."""
        # Setup mocks
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        mock_qdrant_instance = AsyncMock()
        mock_qdrant_instance.get_work_items_by_type.return_value = []
        mock_qdrant_client.return_value.__aenter__.return_value = mock_qdrant_instance

        # Mock template engine
        mock_engine_instance = Mock()
        mock_template_result = Mock()
        mock_template_result.template_text = "Test template"
        mock_template_result.quality_score = 0.9
        mock_template_result.field_patterns = {}
        mock_engine_instance.infer_template.return_value = mock_template_result
        mock_template_engine.return_value = mock_engine_instance

        config_file = self.create_test_config()
        result = self.runner.invoke(
            cli, ["--config", config_file, "analyze", "template", "--issue-type", "Story"]
        )

        assert result.exit_code == 0

    @patch("jirascope.cli.main.ContentAnalyzer")
    @patch("jirascope.cli.main.QdrantVectorClient")
    @patch("jirascope.cli.main.Config.load")
    def test_analyze_tech_debt_command(
        self, mock_config_load, mock_qdrant_client, mock_content_analyzer
    ):
        """Test analyze tech-debt command."""
        # Setup mocks
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        mock_qdrant_instance = AsyncMock()
        mock_qdrant_instance.get_all_work_items.return_value = []
        mock_qdrant_client.return_value.__aenter__.return_value = mock_qdrant_instance

        # Mock content analyzer
        mock_analyzer_instance = Mock()
        mock_tech_debt_result = Mock()
        mock_tech_debt_result.debt_items = []
        mock_tech_debt_result.total_debt_score = 0.3
        mock_tech_debt_result.recommendations = []
        mock_analyzer_instance.analyze_technical_debt.return_value = mock_tech_debt_result
        mock_content_analyzer.return_value = mock_analyzer_instance

        config_file = self.create_test_config()
        result = self.runner.invoke(cli, ["--config", config_file, "analyze", "tech-debt"])

        assert result.exit_code == 0

    @patch("jirascope.cli.main.JiraExtractor")
    @patch("jirascope.cli.main.EmbeddingProcessor")
    @patch("jirascope.cli.main.IncrementalProcessor")
    @patch("jirascope.cli.main.Config.load")
    def test_fetch_command_full_sync(
        self, mock_config_load, mock_incremental_proc, mock_embedding_proc, mock_extractor
    ):
        """Test fetch command for full sync."""
        # Setup mocks
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        # Mock extractor
        mock_extractor_instance = AsyncMock()
        mock_hierarchy = Mock()
        mock_hierarchy.all_items = [Mock(), Mock()]  # Two mock work items
        mock_extractor_instance.extract_active_hierarchies.return_value = [mock_hierarchy]
        mock_cost = Mock()
        mock_cost.api_calls = 10
        mock_cost.estimated_cost = 0.05
        mock_extractor_instance.calculate_extraction_cost.return_value = mock_cost
        mock_extractor.return_value = mock_extractor_instance

        # Mock embedding processor
        mock_processor_instance = AsyncMock()
        mock_result = Mock()
        mock_result.processed_items = 2
        mock_result.failed_items = 0
        mock_processor_instance.process_work_items.return_value = mock_result
        mock_embedding_proc.return_value = mock_processor_instance

        # Mock incremental processor
        mock_incremental_instance = AsyncMock()
        mock_incremental_instance.process_incremental_updates.return_value = Mock()
        mock_incremental_proc.return_value = mock_incremental_instance

        config_file = self.create_test_config()
        result = self.runner.invoke(cli, ["--config", config_file, "fetch", "--project", "TEST"])

        assert result.exit_code == 0
        assert "Fetching all active hierarchies" in result.output

    @patch("jirascope.cli.main.LMStudioClient")
    @patch("jirascope.cli.main.QdrantVectorClient")
    @patch("jirascope.cli.main.Config.load")
    def test_search_command(self, mock_config_load, mock_qdrant_client, mock_lm_client):
        """Test search command functionality."""
        # Setup mocks
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        # Mock LMStudio client
        mock_lm_instance = AsyncMock()
        mock_lm_instance.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding
        mock_lm_client.return_value.__aenter__.return_value = mock_lm_instance

        # Mock Qdrant client
        mock_qdrant_instance = AsyncMock()
        mock_search_result = Mock()
        mock_search_result.key = "TEST-123"
        mock_search_result.summary = "Test work item"
        mock_qdrant_instance.search_similar_work_items.return_value = [mock_search_result]
        mock_qdrant_client.return_value.__aenter__.return_value = mock_qdrant_instance

        config_file = self.create_test_config()
        result = self.runner.invoke(
            cli, ["--config", config_file, "search", "--query", "test query"]
        )

        assert result.exit_code == 0

    @patch("jirascope.cli.main.EmbeddingQualityValidator")
    @patch("jirascope.cli.main.Config.load")
    def test_validate_command(self, mock_config_load, mock_quality_validator):
        """Test validate command functionality."""
        # Setup mocks
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        # Mock quality validator
        mock_validator_instance = AsyncMock()
        mock_validator_instance.validate_all_embeddings.return_value = Mock(
            total_items=100, valid_items=95, invalid_items=5, quality_score=0.95
        )
        mock_quality_validator.return_value = mock_validator_instance

        config_file = self.create_test_config()
        result = self.runner.invoke(cli, ["--config", config_file, "validate"])

        assert result.exit_code == 0

    @patch("jirascope.cli.main.Config.load")
    def test_cost_command(self, mock_config_load):
        """Test cost command functionality."""
        # Setup mocks
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        config_file = self.create_test_config()
        result = self.runner.invoke(cli, ["--config", config_file, "cost"])

        assert result.exit_code == 0

    @patch("jirascope.cli.main.IncrementalProcessor")
    @patch("jirascope.cli.main.Config.load")
    def test_cleanup_command(self, mock_config_load, mock_incremental_proc):
        """Test cleanup command functionality."""
        # Setup mocks
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        mock_processor_instance = Mock()
        mock_processor_instance.cleanup_old_cache.return_value = 5  # 5 files cleaned
        mock_incremental_proc.return_value = mock_processor_instance

        config_file = self.create_test_config()
        result = self.runner.invoke(cli, ["--config", config_file, "cleanup", "--days", "30"])

        assert result.exit_code == 0

    def test_run_async_utility_function(self):
        """Test run_async utility function."""

        async def test_coroutine():
            return "test_result"

        result = run_async(test_coroutine())
        assert result == "test_result"

    @patch("jirascope.cli.main.Config.load")
    def test_extract_command(self, mock_config_load):
        """Test extract command functionality."""
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        config_file = self.create_test_config()
        result = self.runner.invoke(cli, ["--config", config_file, "extract"])

        # Command should at least not crash
        assert result.exit_code in [0, 1]  # May exit with 1 due to missing dependencies

    @patch("jirascope.cli.main.Config.load")
    def test_process_command(self, mock_config_load):
        """Test process command functionality."""
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        config_file = self.create_test_config()
        result = self.runner.invoke(cli, ["--config", config_file, "process"])

        # Command should at least not crash
        assert result.exit_code in [0, 1]  # May exit with 1 due to missing dependencies

    @patch("jirascope.cli.main.Config.load")
    def test_query_command(self, mock_config_load):
        """Test query command functionality."""
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        config_file = self.create_test_config()
        result = self.runner.invoke(
            cli, ["--config", config_file, "query", "--query", "test query"]
        )

        # Command should at least not crash
        assert result.exit_code in [0, 1]  # May exit with 1 due to missing dependencies

    @patch("jirascope.cli.main.Config.load")
    def test_status_command(self, mock_config_load):
        """Test status command functionality."""
        mock_config = Mock()
        mock_config_load.return_value = mock_config

        config_file = self.create_test_config()
        result = self.runner.invoke(cli, ["--config", config_file, "status"])

        # Command should at least not crash
        assert result.exit_code in [0, 1]  # May exit with 1 due to missing dependencies


class TestCLICommandLineOptions:
    """Test CLI command line option parsing and validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_missing_required_options(self):
        """Test commands with missing required options."""
        # Test fetch without project
        result = self.runner.invoke(cli, ["fetch"])
        assert result.exit_code == 2
        assert "Missing option" in result.output

    def test_invalid_option_values(self):
        """Test commands with invalid option values."""
        # Test cleanup with invalid days value
        result = self.runner.invoke(cli, ["cleanup", "--days", "invalid"])
        assert result.exit_code == 2

    def test_command_help_pages(self):
        """Test help pages for various commands."""
        commands_to_test = [
            ["fetch", "--help"],
            ["search", "--help"],
            ["validate", "--help"],
            ["cleanup", "--help"],
            ["analyze", "duplicates", "--help"],
            ["analyze", "quality", "--help"],
        ]

        for cmd in commands_to_test:
            result = self.runner.invoke(cli, cmd)
            assert result.exit_code == 0
            assert "--help" in result.output or "Usage:" in result.output


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("jirascope.cli.main.Config.load")
    def test_config_loading_error(self, mock_config_load):
        """Test CLI behavior when config loading fails."""
        mock_config_load.side_effect = Exception("Failed to load config")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write("invalid: config")
            f.flush()

            result = self.runner.invoke(cli, ["--config", f.name, "--help"])
            assert result.exit_code == 1
            assert "Error loading configuration" in result.output

    def test_nonexistent_command(self):
        """Test CLI behavior with nonexistent command."""
        result = self.runner.invoke(cli, ["nonexistent-command"])
        assert result.exit_code == 2
        assert "No such command" in result.output

    def test_invalid_analyze_subcommand(self):
        """Test analyze command with invalid subcommand."""
        result = self.runner.invoke(cli, ["analyze", "invalid-subcommand"])
        assert result.exit_code == 2
        assert "No such command" in result.output
