"""Tests for the CLI of the `wood_ms train` command."""
import json
import shutil

import numpy as np
from click.testing import CliRunner

from wood_microstructure.cli.main import wood_microstructure


def test_train_command_help():
    """Test the `wood_ms train` command."""
    runner = CliRunner()
    result = runner.invoke(
        wood_microstructure, ['train', '--help']
    )
    assert result.exit_code == 0
    assert 'Usage:' in result.output

def test_train_command_with_invalid_option():
    """Test the `wood_ms train` command with an invalid option."""
    runner = CliRunner()
    result = runner.invoke(
        wood_microstructure, ['train', '--invalid-option']
    )
    assert result.exit_code != 0
    assert 'Error' in result.output

def test_train_command_with_missing_argument():
    """Test the `wood_ms train` command with a missing argument."""
    runner = CliRunner()
    result = runner.invoke(
        wood_microstructure, ['train']
    )
    assert result.exit_code != 0
    assert 'Usage:' in result.output

def test_train_command_without_cross_validation(tmp_path, birch_output_1):
    """Test the `wood_ms train` command with valid arguments."""
    train_dir = tmp_path / 'train'
    test_dir = tmp_path / 'test'
    valid_dir = tmp_path / 'valid'

    shutil.copytree(birch_output_1, train_dir)
    shutil.copytree(birch_output_1, test_dir)
    shutil.copytree(birch_output_1, valid_dir)

    n_epochs = 1

    runner = CliRunner()
    result = runner.invoke(
        wood_microstructure, [
            'train',
            '--train_dir', train_dir.as_posix(),
            '--test_dir', test_dir.as_posix(),
            '--validation_dir', valid_dir.as_posix(),
            '--output_dir', tmp_path.as_posix(),
            '--epochs', str(n_epochs),
        ]
    )
    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    outdir = tmp_path / 'train_surrogate_0'

    assert outdir.exists(), f"Expected output directory {outdir} does not exist."
    assert (outdir / 'best_model.pth').exists(), \
        f"Expected model file {outdir / 'best_model.pth'} does not exist."
    assert (outdir / 'loss_curve_fold_1.dat').exists(), \
        f"Expected loss curve file {outdir / 'loss_curve_fold_1.data'} does not exist."

    params_file = outdir / 'params.json'
    assert params_file.exists(), f"Expected params file {params_file} does not exist."
    with params_file.open('r') as f:
        params = json.load(f)

    assert 'epochs' in params, f"Expected 'epochs' key in params file {params_file} does not exist."
    assert params['epochs'] == n_epochs, f"Expected epochs {n_epochs} does not match params file {params_file}."

def test_train_command_with_cross_validation(tmp_path, birch_output_1):
    """Test the `wood_ms train` command with cross-validation."""
    train_dir = tmp_path / 'train'

    shutil.copytree(birch_output_1, train_dir)

    n_epochs = 1
    n_folds = 2

    runner = CliRunner()
    result = runner.invoke(
        wood_microstructure, [
            'train',
            '--train_dir', train_dir.as_posix(),
            '--output_dir', tmp_path.as_posix(),
            '--epochs', str(n_epochs),
            '--cross_validation',
            '--num_folds', str(n_folds),
        ]
    )
    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    outdir = tmp_path / 'train_surrogate_0'

    assert outdir.exists(), f"Expected output directory {outdir} does not exist."
    for fold in range(1, n_folds + 1):
        fold_model_file = outdir / f"best_model_fold_{fold}.pth"
        fold_loss_curve_file = outdir / f"loss_curve_fold_{fold}.dat"
        assert fold_loss_curve_file.exists(), \
            f"Expected loss curve file {fold_loss_curve_file} does not exist."
        assert fold_model_file.exists(), \
            f"Expected model file {fold_model_file} does not exist."

    assert (outdir / 'params.json').exists(), \
        f"Expected params file {outdir / 'params.json'} does not exist."
    assert (outdir / 'best_model.pth').exists(), \
        f"Expected best model file {outdir / 'best_model.pth'} does not exist."
