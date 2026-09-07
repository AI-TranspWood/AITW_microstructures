import os
import sys

import nrrd
import numpy as np
import numpy.typing as npt

from .main import click, postproc


def input_npy(fpath: str) -> npt.NDArray:
    """Load a numpy file."""
    return np.load(fpath)

def input_nrrd(fpath: str) -> npt.NDArray:
    """Load a nrrd file."""
    return nrrd.read(fpath, index_order='C')[0]

def input_vti(fpath: str) -> npt.NDArray:
    """Load a vti file."""
    try:
        import vtk
    except ImportError:
        sys.exit('ERROR: Install VTK to use VTI files')
    from vtk.util import numpy_support
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(fpath)
    reader.Update()
    vd = reader.GetOutput()

    dims = reader.GetOutput().GetDimensions()

    scalars = vd.GetPointData().GetScalars() or vd.GetPointData().GetArray(0)

    return numpy_support.vtk_to_numpy(scalars).reshape(dims, order='F').astype(np.float32)

def output_npy(fpath: str, data: npt.NDArray):
    """Save a numpy file."""
    np.save(fpath, data)

def output_nrrd(fpath: str, data: npt.NDArray):
    """Save a nrrd file."""
    nrrd.write(fpath, data, index_order='C')

def output_vti(fpath: str, data: npt.NDArray):
    """Save a vti file."""
    try:
        import vtk
    except ImportError:
        sys.exit('ERROR: Install VTK to use VTI files')
    from vtk.util import numpy_support

    data = data.astype(np.uint8)  # Ensure data is uint8 for VTI output
    dims = data.shape
    data_flat = data.flatten(order='F')

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims)
    image_data.SetSpacing(1.0, 1.0, 1.0)
    image_data.SetOrigin(0.0, 0.0, 0.0)

    vtk_array = numpy_support.numpy_to_vtk(data_flat, deep=True, array_type=vtk.VTK_FLOAT)
    image_data.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(fpath)
    writer.SetInputData(image_data)
    writer.Write()

input_funcs = {
    'npy': input_npy,
    'nrrd': input_nrrd,
    'vti': input_vti,
}

output_funcs = {
    'npy': output_npy,
    'nrrd': output_nrrd,
    'vti': output_vti,
}


@postproc.command()
@click.argument('input_file', required=True, type=click.Path(exists=True))
@click.argument('output_file', required=True, type=click.Path())
def volume_convert_format(input_file, output_file):
    """Convert a volume file between formats (npy, nrrd, vti)."""
    input_ext = os.path.splitext(input_file)[1][1:].lower()
    output_ext = os.path.splitext(output_file)[1][1:].lower()

    if input_ext not in input_funcs:
        click.echo(f'Unsupported input file format: {input_ext} (supported: {list(input_funcs.keys())})')
        sys.exit(1)
    if output_ext not in output_funcs:
        click.echo(f'Unsupported output file format: {output_ext} (supported: {list(output_funcs.keys())})')
        sys.exit(1)

    data = input_funcs[input_ext](input_file)
    click.echo(f'Loaded volume data from `{input_file}` with shape {data.shape}')

    output_funcs[output_ext](output_file, data)
    click.echo(f'Saved volume data to `{output_file}`')

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
        click.echo('Please install the package with the extra [utils] dependency to use this feature.')
        sys.exit(1)
    if input_file.endswith('.nrrd'):
        data, header = nrrd.read(input_file, index_order='C')
    elif input_file.endswith('.npy'):
        data = np.load(input_file)
    else:
        click.echo('Unsupported file format. Please provide a .nrrd or .npy file.')
        sys.exit(1)
    click.echo(f'Loaded volume data from `{input_file}` with shape {data.shape}')

    w = data > threshold
    data[w] = 1
    data[~w] = 0
    click.echo(f'Binarized volume data with threshold {threshold}')

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

__all__ = [
    'volume_npy_to_nrrd',
    'plot_volume',
]
