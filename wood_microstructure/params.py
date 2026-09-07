"""Input paramemters"""
import json
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt


class JsonParams:
    params_map: dict[str, str] = {}
    post_set: list[str] = []

    def _to_json(self, data: dict) -> dict:
        """Convert the parameters to a JSON serializable dictionary"""
        return data

    def to_json(self, json_file: str):
        """Save the parameters to a JSON file"""
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        data = self._to_json(data)

        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, json_file: str) -> list[Self]:
        """Create an instance from a JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        res = []
        if isinstance(data, dict):
            res = [cls.from_dict(data)]
        elif isinstance(data, list):
            res = [cls.from_dict(item) for item in data]
        else:
            raise ValueError('Invalid data format in JSON file')
        return res

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create an instance from a JSON file"""
        post = {}
        data = {cls.params_map.get(k, k): v for k, v in data.items()}
        for k in cls.post_set:
            if k in data:
                post[k] = data.pop(k)
        res = cls(**data)
        for k, v in post.items():
            setattr(res, k, v)
        return res

@dataclass
class BaseParams(JsonParams):
    """Base class for parameters"""
    period_parameter: int  # This parameter is related to the period of the year ring size

    cell_r: float  # The grid distance for the nodes we generated. In Unit of voxels.
    cell_length: float  # Average fiber length
    cell_length_variance: float  # Standard deviation of fiber length
    cell_wall_thick: float  # Cell wall thickness
    cell_end_thick: int  # End of cell wall thickness along L direction

    ray_height: float  # The width of the ray cell
    ray_space: float  # The space between ray cell along T direction. The distance is raySpace*cellR
    ray_cell_length: float  # Ray cell length along radial direction
    ray_cell_variance: float  # Ray cell length deviation along radial direction
    ray_cell_num: float  # Ray cell count in a group
    ray_cell_num_std: float  # Ray cell count in a group

    vessel_length: float  # Average vessel length
    vessel_length_variance: float  # Standard deviation of fiber length
    vessel_thicker: int  # Assume vessel is thicker than ray cells
    vessel_count: int  # This number is used to control the vessel number and distribution.

    is_exist_vessel: bool = True  # Whether to generate vessel cells
    is_exist_ray_cell: bool = True  # Whether to generate ray cells

    random_seed: int = 42  # Random seed initialization for reproducibility

    size_volume: tuple[int, int, int] = (500, 500, 200)  # The size of the volume to be generated
    extra_size: tuple[int, int, int] = (150, 200, 100)  # Extra size for the enlarged image
    slice_interest_space: int = 100  # We generate one slice every XXX slices to add random noise before interpolation.

    apply_local_deform: bool = True  # Whether to apply local deformation
    apply_global_deform: bool = True  # Whether to apply global deformation

    save_slices_as_2d: bool = True
    save_volume_as_3d: bool = True
    save_volume_format: str = 'nrrd'
    save_local_dist: bool = True
    save_global_dist: bool = True
    # write_local_deform_data: bool = True
    # write_global_deform_data: bool = False

    surrogate: bool = False  # Whether to use surrogate model for local deformation
    binarize_threshold: int = None  # Threshold for binarization of the final volume data. If None, no bin is applied.

    # Not user defined
    neighbor_local = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]], dtype=int)  # d-indices of the neighbor grid nodes

    # Internal parameters
    _all_slices = False  # Whether to save all slices or not
    _size_im_enlarge: tuple[int, int, int] = None
    _x_vector: npt.NDArray = None
    _y_vector: npt.NDArray = None
    _grid: tuple[npt.NDArray, npt.NDArray] = None
    _num_grid_nodes: int = None

    _save_slice: list[int] | str = None  # List of slices (Z-index) to save (NOTE: inputfile is 1-indexed)
    _save_slice_map: dict[int, int] = None

    params_map = {
        'sizeVolume': 'size_volume',
        'saveSlice': 'save_slice',
        'cellR': 'cell_r',
        'cellLength': 'cell_length',
        'cellLengthVariance': 'cell_length_variance',
        'rayCellLength': 'ray_cell_length',
        'rayCell_variance': 'ray_cell_variance',
        'rayCellNum': 'ray_cell_num',
        'rayCellNumStd': 'ray_cell_num_std',
        'vesselLength': 'vessel_length',
        'vesselLengthVariance': 'vessel_length_variance',
        'raywHeight': 'ray_height',
        'raySpace': 'ray_space',
        'isExistVessel': 'is_exist_vessel',
        'isExistRayCell': 'is_exist_ray_cell',
        'cellWallThick': 'cell_wall_thick',
        'writeGlobalDeformData': 'save_global_dist',
        'writeLocalDeformData': 'save_local_dist',
    }

    post_set = ['save_slice']

    @property
    def save_slice(self):
        """List of slices (Z-index) to save (NOTE: inputfile is 1-indexed)"""
        if self._save_slice is None:
            self.save_slice = 'all'  # Default to save all slices
        return self._save_slice

    @save_slice.setter
    def save_slice(self, value: list[int] | str):
        self._all_slices = False
        if isinstance(value, str):
            if value == 'all':
                self._all_slices = True
                value = tuple(range(self.size_im_enlarge[2]))
            else:
                value = (int(value) - 1,)
        elif isinstance(value, int):
            value = (value - 1,)
        elif isinstance(value, (tuple, list)):
            value = tuple(int(s) - 1 for s in value)

        self._save_slice = value

    @property
    def all_slices(self):
        """Whether to save all slices"""
        return self._all_slices

    @property
    def save_slice_map(self):
        """Map of saved slices"""
        if self._save_slice_map is None:
            self._save_slice_map = {s: i for i, s in enumerate(self.save_slice)}
        return self._save_slice_map

    @property
    def size_im_enlarge(self):
        """Size of enlarged image"""
        if self._size_im_enlarge is None:
            self._size_im_enlarge = np.array(self.size_volume) + np.array(self.extra_size)
        return self._size_im_enlarge

    @property
    def x_vector(self):
        """X vector"""
        if self._x_vector is None:
            self._x_vector = np.arange(5, self.size_im_enlarge[0] - 4, self.cell_r)  # Right-inclusive
        return self._x_vector

    @property
    def y_vector(self):
        """Y vector"""
        if self._y_vector is None:
            self._y_vector = np.arange(5, self.size_im_enlarge[1] - 4, self.cell_r)  # Right-inclusive
        return self._y_vector

    @property
    def grid(self):
        """Tuple of X and Y  2D grid"""
        if self._grid is None:
            self._grid = np.meshgrid(self.x_vector, self.y_vector, indexing='ij')
        return self._grid

    @property
    def x_grid(self):
        """X coordinate of the 2D grid (slice of 3D)"""
        return self.grid[0]

    @property
    def y_grid(self):
        """Y coordinate of the 2D grid (slice of 3D)"""
        return self.grid[1]

    @property
    def num_grid_nodes(self):
        """Number of grid nodes"""
        if self._num_grid_nodes is None:
            self._num_grid_nodes = self.x_grid.size()
        return self._num_grid_nodes

    def _to_json(self, data: dict) -> dict:
        """Convert the parameters to a JSON serializable dictionary"""
        if self.all_slices:
            data['saveSlice'] = 'all'
        else:
            data['saveSlice'] = [s + 1 for s in self.save_slice]
        return data

@dataclass
class BirchParams(BaseParams):
    """Define the parameters for birch"""
    period_parameter: int = 0

    cell_r: float = 14.5
    cell_length: float = 2341
    cell_length_variance: float = 581
    cell_wall_thick: float = 2
    cell_end_thick: int = 4

    ray_height: float = 42
    ray_space: float = 20
    ray_cell_length: float = 62
    ray_cell_variance: float = 15
    ray_cell_num: float = 11.33
    ray_cell_num_std: float = 3.39

    vessel_length: float = 780
    vessel_length_variance: float = 195
    vessel_thicker: int = 1
    vessel_count: int = 50

    is_exist_vessel: bool = True
    is_exist_ray_cell: bool = True


@dataclass
class SpruceParams(BaseParams):
    """Define the parameters for spruce"""
    period_parameter: int = 1000

    cell_r: float = 14.5
    cell_length: float = 4877
    cell_length_variance: float = 1219
    cell_wall_thick: float = 3
    cell_end_thick: int = 2

    ray_height: float = 40
    ray_space: float = 0
    ray_cell_length: float = 149.4
    ray_cell_variance: float = 38.5
    ray_cell_num: float = 8.44
    ray_cell_num_std: float = 4.39

    vessel_length: float = 4877
    vessel_length_variance: float = 1219
    vessel_thicker: int = 0
    vessel_count: int = 0

    is_exist_vessel: bool = True
    is_exist_ray_cell: bool = True

@dataclass
class FitPorosityParams(JsonParams):
    """Define the parameters for fitting porosity"""
    input_file: str = None  # Input file path to a volume data file.

    threshold: float = 127  # Threshold for binarization of the final volume data. If None, no bin is applied.
    solid_is_high: bool = True  # Whether the solid part is dark or light in the image.
                              # If True, the solid part is dark and the void part is light.
                              # If False, the solid part is light and the void part is dark.
    down: int = 2  # Downsample factor (1=no downsampling)

    smooth_low_iters: int = 0  # Number of iterations for low-pass smoothing. If <= 0, no smoothing is applied.
    smooth_sigma: float = 0.6  # Sigma for Gaussian smoothing. If <= 0, no smoothing is applied.

    thicken: int = 0  # Dilate solid by N voxels (thicken walls, narrow throats)
    final_smooth_iters: int = 0  # Number of iterations for final low-pass smoothing. If <= 0, no smoothing is applied.
    final_smooth_sigma: float = 0.5  # Sigma for final Gaussian smoothing. If <= 0, no smoothing is applied.

    upsample_intermediate: float = None  # Supersampling: first upsample by this factor (e.g., 3)
    upsample_final: float = None  # Supersampling: then downsample to this net factor (e.g., 2). Creates smoother upsampling via anti-aliasing.

    pad: int = 0  # Add N voxels of pore padding (positive) or crop N voxels (negative). Example: --pad 10 adds padding, --pad -5 crops 5 voxels from edges.
    pad_xy_only: bool = False

    porosity: float = None  # Target porosity [0, 1]. Uses SDF threshold search on raw grayscale. Pass 'None' or omit to disable (use --threshold instead).
    sdf_sigma: float = 0.5  # SDF Gaussian blur sigma (higher = more boundary smearing)
    adjust_porosity_post: bool = False  # Fine-tune porosity after processing via post-processing SDF adjustment

    majority_filter: int = 1  # Remove thin protrusions (iterations). 0=off, 2-3 for aggressive filtering.
