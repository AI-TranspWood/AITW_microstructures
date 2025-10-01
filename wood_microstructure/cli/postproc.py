import os
import sys

import numpy as np

from .main import click, postproc


@postproc.command()
@click.argument('input_file', required=True, type=click.Path(exists=True))
@click.option(
    '--threshold',
    type=click.IntRange(0, 255),
    default=None,
    help='Threshold value for binarization (0-255)',
    )
def convert_volume_to_nrrd(input_file, threshold):
    """Convert a numpy volume file to nrrd format."""
    try:
        import nrrd
    except ImportError:
        print('Please install the package with the extra [utils] dependency to use this feature.')
        sys.exit(1)
    dirname = os.path.dirname(input_file)
    name, ext = os.path.splitext(os.path.basename(input_file))
    outfile = os.path.join(dirname, f'{name}.nrrd')

    data = np.ascontiguousarray(np.load(input_file))
    print(f'Loaded volume data from `{input_file}` with shape {data.shape}')

    if threshold is not None:
        w = data > threshold
        data[w] = 1
        data[~w] = 0
        print(f'Binarized volume data with threshold {threshold}')

    nrrd.write(
        outfile,
        data,
        index_order='C',
    )
    print(f'Saved nrrd file to `{outfile}`')

@postproc.command()
@click.argument('input_file', required=True, type=click.Path(exists=True))
@click.option(
    '--threshold',
    type=click.IntRange(0, 255),
    default=125,
    help='Threshold value for binarization (0-255)',
)
def plot_volume(input_file, threshold):
    """Plot a numpy volume file using mayavi."""
    try:
        import matplotlib.pyplot as plt
        from mayavi import mlab
        from tvtk.util import ctf
    except ImportError:
        print('Please install the package with the extra [utils] dependency to use this feature.')
        sys.exit(1)
    data = np.load(input_file)
    print(f'Loaded volume data from `{input_file}` with shape {data.shape}')

    w = data > threshold
    data[w] = 1
    data[~w] = 0
    print(f'Binarized volume data with threshold {threshold}')

    mlab.figure(bgcolor=(1.0, 1.0, 1.0), size=(1600, 1600))
    src = mlab.pipeline.scalar_field(data)
    src.update_image_data = True
    volume = mlab.pipeline.volume(src, vmin=0, vmax=1)

    c = ctf.save_ctfs(volume._volume_property)
    c['rgb'] = plt.get_cmap('gray')(np.arange(2))
    ctf.load_ctfs(c, volume._volume_property)

    volume.update_ctf = True

    mlab.axes()
    mlab.show()
