"""Tests for the CLI of the `wood_ms generate` command."""
import json

from click.testing import CliRunner

from wood_microstructure.cli.main import wood_microstructure


def test_generate_command_help():
    """Test the `wood_ms generate` command."""
    runner = CliRunner()
    result = runner.invoke(
        wood_microstructure, ['generate', '--help']
    )
    assert result.exit_code == 0
    assert 'Usage:' in result.output

def test_generate_command_with_invalid_option():
    """Test the `wood_ms generate` command with an invalid option."""
    runner = CliRunner()
    result = runner.invoke(
        wood_microstructure, ['generate', '--invalid-option']
    )
    assert result.exit_code != 0
    assert 'Error' in result.output

def test_generate_command_with_missing_argument():
    """Test the `wood_ms generate` command with a missing argument."""
    runner = CliRunner()
    result = runner.invoke(
        wood_microstructure, ['generate']
    )
    assert result.exit_code != 0
    assert 'Error' in result.output

def test_generate_command_birch_1(birch_output_1):
    """Test the `wood_ms generate` command with valid arguments."""
    file_out = birch_output_1 / 'params.json'

    assert birch_output_1.exists(), f"Expected output directory {birch_output_1} does not exist."
    assert file_out.exists(), f"Expected output file {file_out} does not exist."

    with open(file_out, 'r') as f:
        params = json.load(f)

    assert params['random_seed'] == 123, 'Random seed in output params does not match the provided seed.'
    assert params['surrogate'] is False, "Expected 'surrate' to be False in the output params."

def test_generate_command_birch_2(birch_output_2):
    """Test the `wood_ms generate` command with valid arguments."""
    file_out = birch_output_2 / 'params.json'

    assert birch_output_2.exists(), f"Expected output directory {birch_output_2} does not exist."
    assert file_out.exists(), f"Expected output file {file_out} does not exist."

    with open(file_out, 'r') as f:
        params = json.load(f)

    assert params['surrogate'] is True, "Expected 'surrogate' to be True in the output params."
    assert params['fit_porosity'] is True, "Expected 'fit_porosity' to be True in the output params."

    fit_porosity_outdir = birch_output_2 / 'PorosityFitting' / 'fit_porosity_0'
    fit_porosity_outfile = fit_porosity_outdir / 'fit_porosity_volume.vti'

    assert fit_porosity_outdir.exists(), f"Expected output directory {fit_porosity_outdir} does not exist."
    assert fit_porosity_outfile.exists(), f"Expected output file {fit_porosity_outfile} does not exist."

def test_generate_command_spruce_1(spruce_output_1):
    """Test the `wood_ms generate` command with valid arguments."""
    file_out = spruce_output_1 / 'params.json'

    assert spruce_output_1.exists(), f"Expected output directory {spruce_output_1} does not exist."
    assert file_out.exists(), f"Expected output file {file_out} does not exist."

    with open(file_out, 'r') as f:
        params = json.load(f)
    assert params['random_seed'] == 456, 'Random seed in output params does not match the provided seed.'
    assert params['surrogate'] is False, "Expected 'surrogate' to be False in the output params."
