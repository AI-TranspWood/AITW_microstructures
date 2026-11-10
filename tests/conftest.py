"""Fixtures for the aitw-viscosity-surrogate tests."""
import json

import pytest
from click.testing import CliRunner


def dict_to_cli_args(params_dict):
    """Convert a dictionary of parameters to a list of CLI arguments."""
    cli_args = []
    for key, value in params_dict.items():
        if isinstance(value, bool):
            if value:
                cli_args.append(f"--{key}")
        else:
            cli_args.extend([f"--{key}", str(value)])
    return cli_args

@pytest.fixture(scope='session')
def global_tmp_dir(tmp_path_factory):
    """Fixture to provide a temporary directory for the entire test session."""
    return tmp_path_factory.mktemp('aitw_viscosity_surrogate_tests')

@pytest.fixture(scope='session')
def small_input_path():
    """Fixture to provide the path to a small birch input file."""
    return 'examples/test.json'

@pytest.fixture(scope='session')
def fit_porosity_input_path():
    """Fixture to provide the path to a fit porosity input file."""
    return 'examples/fit_porosity.json'

@pytest.fixture(scope='session')
def birch_run_params_1():
    """Fixture to provide the parameters for the first birch run."""
    return {
        'random_seed': 123,
    }

@pytest.fixture(scope='session')
def birch_run_params_2(fit_porosity_input_path):
    """Fixture to provide the parameters for the second birch run."""
    return {
        'surrogate': True,
        'fit_porosity': True,
        'fit_porosity_config': fit_porosity_input_path,
    }

@pytest.fixture(scope='session')
def spruce_run_params_1():
    """Fixture to provide the parameters for the spruce run."""
    return {
        'random_seed': 456,
    }

@pytest.fixture(scope='session')
def birch_output_1(birch_run_params_1, global_tmp_dir, small_input_path):
    """Fixture to provide the path to the first birch output directory."""
    from wood_microstructure.cli.main import wood_microstructure

    extra_params = dict_to_cli_args(birch_run_params_1)

    runner = CliRunner()
    outdir = global_tmp_dir / 'birch_output_1'
    result = runner.invoke(
        wood_microstructure, [
            'generate', 'birch',
            '--config-file', small_input_path, '--output_dir', outdir,
            *extra_params
        ]
    )
    assert result.exit_code == 0, f"Command failed with output: {result.output}"
    return outdir / 'SaveBirch_0'

@pytest.fixture(scope='session')
def birch_output_2(birch_run_params_2, global_tmp_dir, small_input_path):
    """Fixture to provide the path to the second birch output directory."""
    from wood_microstructure.cli.main import wood_microstructure

    extra_params = dict_to_cli_args(birch_run_params_2)

    runner = CliRunner()
    outdir = global_tmp_dir / 'birch_output_2'
    result = runner.invoke(
        wood_microstructure, [
            'generate', 'birch',
            '--config-file', small_input_path, '--output_dir', outdir,
            *extra_params
        ]
    )
    assert result.exit_code == 0, f"Command failed with output: {result.output}"
    return outdir / 'SaveBirch_0'

@pytest.fixture(scope='session')
def spruce_output_1(spruce_run_params_1, global_tmp_dir, small_input_path):
    """Fixture to provide the path to the spruce output directory."""
    from wood_microstructure.cli.main import wood_microstructure

    extra_params = dict_to_cli_args(spruce_run_params_1)

    runner = CliRunner()
    outdir = global_tmp_dir / 'spruce_output_1'
    result = runner.invoke(
        wood_microstructure, [
            'generate', 'spruce',
            '--config-file', small_input_path, '--output_dir', outdir,
            *extra_params
        ]
    )
    assert result.exit_code == 0, f"Command failed with output: {result.output}"
    return outdir / 'SaveSpruce_0'
