from click.testing import CliRunner
from sps_pipeline.pipeline import main
import re


def test_command_line_interface():
    """Check the output when no commands are invoked"""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert (
        "Runner script for the Slow Pulsar Search prototype pipeline" in result.output
    )
    assert re.match(
        r".*-h, --help\s+Show this message and exit.", result.output, re.DOTALL
    )
