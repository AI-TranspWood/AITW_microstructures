"""Tests for the CLI of the `wood_ms postproc` command."""
import json

import numpy as np
from click.testing import CliRunner

from wood_microstructure.cli.main import wood_microstructure


def test_postproc_command_help():
    """Test the `wood_ms postproc` command."""
    runner = CliRunner()
    result = runner.invoke(
        wood_microstructure, ['postproc', '--help']
    )
    assert result.exit_code == 0
    assert 'Usage:' in result.output

def test_postproc_command_with_invalid_option():
    """Test the `wood_ms postproc` command with an invalid option."""
    runner = CliRunner()
    result = runner.invoke(
        wood_microstructure, ['postproc', '--invalid-option']
    )
    assert result.exit_code != 0
    assert 'Error' in result.output

def test_postproc_command_volume_convert(tmp_path, birch_output_1):
    """Test the `wood_ms postproc volume_convert` command."""
    input_file = birch_output_1 / 'FinalVolume3D' / 'FinalVolume.nrrd'
    output_file = tmp_path / 'FinalVolume.npy'

    runner = CliRunner()
    result = runner.invoke(
        wood_microstructure, [
            'postproc', 'volume-convert-format',
            input_file.as_posix(), output_file.as_posix(),
        ]
    )
    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    assert output_file.exists(), f"Expected output file {output_file} does not exist."

    with (birch_output_1 / 'params.json').open('r') as f:
        params = json.load(f)

    shape_expected = tuple(params['size_volume'])

    # Load the converted volume and check its properties
    converted_volume = np.load(output_file)
    assert converted_volume.shape == shape_expected, \
        f"Converted volume shape {converted_volume.shape} does not match expected shape {shape_expected}."

def test_postproc_command_filter_porosity(tmp_path, birch_output_1, fit_porosity_input_path):
    """Test the `wood_ms postproc filter-porosity` command."""
    input_file = birch_output_1 / 'FinalVolume3D' / 'FinalVolume.nrrd'

    runner = CliRunner()
    result = runner.invoke(
        wood_microstructure, [
            'postproc', 'filter-porosity',
            '--config-file', fit_porosity_input_path,
            '--input_file', input_file.as_posix(),
            '--output_dir', tmp_path.as_posix(),
        ]
    )
    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    output_dir = tmp_path / 'fit_porosity_0'
    output_file = output_dir / 'fit_porosity_volume.vti'

    assert output_dir.exists(), f"Expected output directory {output_dir} does not exist."
    assert output_file.exists(), f"Expected output file {output_file} does not exist."
