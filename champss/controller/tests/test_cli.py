import re

import controller
from click.testing import CliRunner


def test_command_line_interface():
    """Check the output when no commands are invoked."""
    runner = CliRunner()
    result = runner.invoke(controller.cli, ["--help"])
    assert result.exit_code == 0
    assert "L1 controller for Slow Pulsar Search" in result.output
    assert re.match(
        r".*-h, --help\s+Show this message and exit.", result.output, re.DOTALL
    )
