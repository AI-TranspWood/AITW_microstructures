"""Ray cells"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline


def MATLAB_definition():
    # % function to generate ray cells
    # rayCellLength  = Params.rayCellLength;
    # rayCell_variance  = Params.rayCell_variance;
    # rayHeight       = Params.rayHeight;
    # xGrid_all      = Params.xGrid_all;
    # yGrid_all      = Params.yGrid_all;
    # Thickness_all  = Params.Thickness_all;
    # sizeImEnlarge  = Params.sizeImEnlarge;
    # xGrid          = Params.xGrid;
    # cellEndThick   = Params.cellEndThick;
    # cellThick      = Params.cellThick;

    # % raycellWidth(raycellWidth<-rayHeight | raycellWidth>sizeImEnlarge(3)+rayHeight) = [];
    # for i_raycolumn = raycellXind
    #     % between first column and second column need a deviation
    #     raycolumn_rand = 1/2*rayHeight;
    #     s = 1;
    #     vesselEndLoc = [];
    #     vesselEndLoc(1) = -rayCellLength*rand(1);
    #     while vesselEndLoc(s)<sizeImEnlarge(1)+rayCellLength
    #         s = s+1;
    #         temp = rayCellLength + rayCell_variance*randn(1);
    #         if temp<rayCellLength/3
    #             temp=2*rayCellLength;
    #         else
    #             if temp>2*rayCellLength
    #                 temp=2*rayCellLength;
    #             end
    #         end
    #         vesselEndLoc(s) = vesselEndLoc(s-1)+temp;
    #     end
    #     vesselEndLoc = round(vesselEndLoc);
    #     vesselEndLoc(vesselEndLoc>sizeImEnlarge(1)+rayCellLength/2) = [];
    #     for i_rayrow = 1:length(vesselEndLoc)-1
    #         m2 = 0;
    #         for j_slice = round(raycellWidth)
    #             m2 = m2+1;

    #             if mod(i_raycolumn,2) == 0
    #                 k = round(j_slice+round(raycolumn_rand));
    #             else
    #                 k = j_slice;
    #             end
    #             t              = round((max(1,k)+min(k+rayHeight,sizeImEnlarge(3)))/2);
    #             if t >= 1 && t<sizeImEnlarge(3)-10
    #             xGrid_t        = reshape(xGrid_all(:,t),size(xGrid));
    #             yGrid_t        = reshape(yGrid_all(:,t),size(xGrid));
    #             thickGrid_t    = reshape(Thickness_all(:,t),size(xGrid));

    #             yInterp1_C     = spline(xGrid_t(:,i_raycolumn),yGrid_t(:,i_raycolumn),1:sizeImEnlarge(1))-1.5;
    #             thickInterp_C  = spline(xGrid_t(:,i_raycolumn),thickGrid_t(:,i_raycolumn),1:sizeImEnlarge(1));

    #             yInterp2_C     = spline(xGrid_t(:,i_raycolumn+1),yGrid_t(:,i_raycolumn+1),1:sizeImEnlarge(1))+1.5;

    #             cell_center    = [(1:length(yInterp2_C))',round(yInterp2_C(:)+yInterp1_C(:))/2,...
    #                     repmat((max(1,k)+min(k+rayHeight,sizeImEnlarge(3)))/2,length(yInterp2_C),1)];
    #             cell_r         = [(yInterp2_C(:)-yInterp1_C(:))/2,repmat((min(k+rayHeight,sizeImEnlarge(3))-max(1,k))/2,length(yInterp2_C),1)]+0.5;

    #             vesselEndLoc_column = vesselEndLoc+round(mod(m2,2)*rayCellLength/2);
    #             if vesselEndLoc_column(i_rayrow+1)<=sizeImEnlarge(1)-1 && vesselEndLoc_column(i_rayrow)>= 2



    #                 cell_neighPt   = [vesselEndLoc_column(i_rayrow),vesselEndLoc_column(i_rayrow+1);
    #                                    round(yInterp1_C(vesselEndLoc_column(i_rayrow))),...
    #                                    round(yInterp2_C(vesselEndLoc_column(i_rayrow)));...
    #                                    max(1,k),min(k+rayHeight,sizeImEnlarge(3))];

    #                 if i_rayrow == 1
    #                     i_valid = vesselEndLoc_column(i_rayrow)+cellEndThick:vesselEndLoc_column(i_rayrow+1)-cellEndThick/2;

    #                 else
    #                     if i_rayrow == length(vesselEndLoc)-1
    #                         i_valid = vesselEndLoc_column(i_rayrow)+cellEndThick/2-1:vesselEndLoc_column(i_rayrow+1)-cellEndThick/2;
    #                     else
    #                         i_valid = vesselEndLoc_column(i_rayrow)+cellEndThick/2-1:vesselEndLoc_column(i_rayrow+1)-cellEndThick/2;
    #                     end
    #                 end


    #                 for i = vesselEndLoc_column(i_rayrow)+1:vesselEndLoc_column(i_rayrow+1)

    #                     if j_slice == min(raycellWidth)
    #                     volImgRef_final(i,...
    #                                 yInterp1_C(i):yInterp2_C(i),...
    #                                 cell_center(i,3):cell_neighPt(3,2)) = 255;
    #                     else
    #                         if j_slice == max(raycellWidth)
    #                             volImgRef_final(i,...
    #                                         yInterp1_C(i):yInterp2_C(i),...
    #                                         cell_neighPt(3,1):cell_center(i,3)) = 255;
    #                         else
    #                             volImgRef_final(i,...
    #                                         yInterp1_C(i):yInterp2_C(i),...
    #                                         cell_neighPt(3,1):cell_neighPt(3,2)) = 255;
    #                         end
    #                     end

    #                     if find(round(i_valid) == i)

    #                         for j = cell_neighPt(2,1):cell_neighPt(2,2)
    #                             for s = cell_neighPt(3,1):cell_neighPt(3,2)
    #                                 inner_elipse = (j-cell_center(i,2)).^2./(cell_r(i,1)-thickInterp_C(i))^2 ...
    #                                             + (s-cell_center(i,3)).^2./(cell_r(i,2)-thickInterp_C(i)).^2;

    #                                 outer_elipse = (j-cell_center(i,2)).^2./(cell_r(i,1)).^2 ...
    #                                             + (s-cell_center(i,3)).^2./(cell_r(i,2)).^2;

    #                                 if outer_elipse<1
    #                                     volImgRef_final(i,j,s) = uint8(1./(1+exp(-(inner_elipse-1)/0.05))*255);
    #                                 end

    #                             end
    #                         end

    #                     end
    #                 end

    #             end
    #             end
    #         end
    #     end
    # end


    pass

@dataclass
class RayCellParams:
    """Ray cell parameters"""
    random_seed: int = 42

    size_volume: Iterable[int, int, int] = (500, 1200, 300)

    cell_r: float = 14.5
    cell_length: float = 2341
    cell_length_variance: float = 581
    cell_wall_thick: float = 2

    ray_height: float = 42
    ray_space: float = 20

    ray_cell_length: float = 62
    ray_cell_variance: float = 15
    ray_cell_num: float = 11.33
    ray_cell_num_std: float = 3.39

    vessel_length: float = 780
    vessel_length_variance: float = 195

    is_exist_vessel: bool = True
    is_exist_ray_cell: bool = True

    save_slice: bool = True
    save_volume_as_3d: bool = True
    write_local_deform_data: bool = True
    write_global_deform_data: bool = False

    # Not user defined
    slice_interest_space: int = 100
    cell_end_thick: float = 4
    neighbor_local: npt.NDArray = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])
    vessel_count: int = 50

    _size_im_enlarge: Iterable[int, int, int] = None
    _x_vector: npt.NDArray = None
    _y_vector: npt.NDArray = None
    _grid: tuple[npt.NDArray, npt.NDArray] = None
    _num_grid_nodes: int = None

    @property
    def size_im(self):
        """Size of image"""
        return np.array(self.size_volume)

    @property
    def size_im_enlarge(self):
        """Size of enlarged image"""
        if self._size_im_enlarge is None:
            extra_sz = (150, 200, 100)
            self._size_im_enlarge = np.array(self.size_volume) + np.array(extra_sz)
        return self._size_im_enlarge

    @property
    def x_vector(self):
        """X vector"""
        if self._x_vector is None:
            self._x_vector = np.arange(5, self.size_im_enlarge[0] - 5, self.cell_r)
        return self._x_vector

    @property
    def y_vector(self):
        """Y vector"""
        if self._y_vector is None:
            self._y_vector = np.arange(5, self.size_im_enlarge[1] - 5, self.cell_r)
        return self._y_vector

    @property
    def grid(self):
        """X grid"""
        if self._grid is None:
            self._grid = np.meshgrid(self.x_vector, self.y_vector, indexing='ij')
        return self._grid

    @property
    def x_grid(self):
        """X grid"""
        return self.grid[0]

    @property
    def y_grid(self):
        """Y grid"""
        return self.grid[1]

    @property
    def num_grid_nodes(self):
        """Number of grid nodes"""
        if self._num_grid_nodes is None:
            self._num_grid_nodes = self.x_grid.size()
        return self._num_grid_nodes


class RayCell:
    def __init__(self, params: RayCellParams):
        self.params = params

        self.x_grid_all = None
        self.y_grid_all = None
        self.thickness_all = None

    def get_grid_all(self):
        """Specify the location of grid nodes and the thickness (with disturbance)"""
        slice_interest = np.arange(1, self.params.size_im_enlarge[2], self.params.slice_interest_space)
        slice_interest = np.append(slice_interest, self.params.size_im_enlarge[2])

        l = len(slice_interest)

        grid_shape = self.params.x_grid.shape

        x_grid = self.params.x_grid#.flatten()
        y_grid = self.params.y_grid#.flatten()

        # x_grid_interp = np.empty((x_grid.size, len(slice_interest)))
        # y_grid_interp = np.empty((y_grid.size, len(slice_interest)))
        # thickness_interp = np.empty((x_grid.size, len(slice_interest)))
        # for t, i_slice in enumerate(slice_interest):
        #     x_grid_interp[:, t] = x_grid + np.random.randn(*grid_shape, 1) * 3 - 1.5
        #     y_grid_interp[:, t] = y_grid + np.random.randn(*grid_shape, 1) * 3 - 1.5
        #     thickness_interp[:, t] = self.params.cell_wall_thick - 0.5 + np.random.randn(*grid_shape, 1)
        x_grid_interp = np.random.rand(*grid_shape, l) * 3 - 1.5 + x_grid
        y_grid_interp = np.random.rand(*grid_shape, l) * 3 - 1.5 + y_grid
        thickness_interp = np.random.rand(*grid_shape, l) + self.params.cell_wall_thick - 0.5

        # TODO: redo this with proper broadcasting and use RegularGridInterpolator
        x_grid_interp = x_grid_interp.reshape(-1, l)
        y_grid_interp = y_grid_interp.reshape(-1, l)
        thickness_interp = thickness_interp.reshape(-1, l)

        interp_x = np.arange(1, self.params.size_im_enlarge[2] + 1)
        x_grid_all = np.empty_like(x_grid_interp)
        y_grid_all = np.empty_like(y_grid_interp)
        thickness_all = np.empty_like(thickness_interp)
        for i in range(x_grid.size):
            cs_x = CubicSpline(slice_interest, x_grid_interp[i, :])
            cs_y = CubicSpline(slice_interest, y_grid_interp[i, :])
            cs_t = CubicSpline(slice_interest, thickness_interp[i, :])
            x_grid_all[i, :] = cs_x(interp_x)
            y_grid_all[i, :] = cs_y(interp_x)
            thickness_all[i, :] = cs_t(interp_x)

        self.x_grid_all = x_grid_all
        self.y_grid_all = y_grid_all
        self.thickness_all = thickness_all

        return x_grid_all, y_grid_all, thickness_all

    def get_ray_cell_indexes(self) -> npt.NDArray:
        """Get ray cell indexes"""
        ray_cell_x_ind_all = np.empty((1, 0))
        if self.params.is_exist_ray_cell:
            ray_cell_linspace = np.arange(10, len(self.params.y_vector) - 9, self.params.ray_space)
            ray_cell_x_ind_all = ray_cell_linspace + np.random.rand(len(ray_cell_linspace)) * 10 - 5
            ray_cell_x_ind_all = ray_cell_x_ind_all // 2 + 1

        return ray_cell_x_ind_all


    def get_vessels_all(self, ray_cell_x_ind_all: npt.NDArray = None):
        """Get vessels"""
        if not self.params.is_exist_vessel:
            return np.empty((0, 3))
        x_vector = self.params.x_vector
        y_vector = self.params.y_vector

        x_rand_1 = np.round(np.random.rand(self.params.vessel_count) * (len(x_vector) - 16) + 8) / 2 * 2 - 1
        y_rand_1 = np.round(np.random.rand(self.params.vessel_count) * (len(y_vector) - 14) + 7) / 4 * 4
        y_rand_2 = np.round(np.random.rand(self.params.vessel_count // 2) * (len(y_vector) - 14) + 7) / 2 * 2
        x_rand_2 = np.round(np.random.rand(self.params.vessel_count // 2) * (len(x_vector) - 16) + 8) / 2 * 2

        x_rand_all = np.concatenate((x_rand_1, x_rand_2))
        y_rand_all = np.concatenate((y_rand_1, y_rand_2))
        vessel_all = np.column_stack((x_rand_all, y_rand_all))

        # Remove some vessel that too close to the other vessels
        vessel_all = self.vessel_filter_close(vessel_all)
        vessel_all = self.vessel_filter_ray_close(vessel_all, ray_cell_x_ind_all)
        vessel_all = self.vessel_extend(vessel_all)
        vessel_all = self.vessel_filter_ray_close(vessel_all, ray_cell_x_ind_all)

        return vessel_all

    def vessel_filter_close(self, vessel_all: npt.NDArray):
        """Filter the vessel that are too close to the other vessels"""
        all_idx = set()
        done = set()
        for i, vessel in enumerate(vessel_all):
            if vessel[0] == 0:
                continue
            dist = np.abs(vessel_all - vessel)
            mark0 = np.where((dist[:, 0] <= 6) & (dist[:, 1] <= 4))[0]
            if i not in done:
                all_idx.add(i)
            done.update(mark0)

        return vessel_all[list(all_idx)]

    def vessel_filter_ray_close(self, vessel_all: npt.NDArray, ray_cell_x_ind_all: npt.NDArray):
        """Filter the vessel that are too close to the ray cells"""
        all_idx = set()
        for i, vessel in enumerate(vessel_all):
            diff = vessel[0] - ray_cell_x_ind_all
            if not np.any((diff >= -3) & (diff <= 4)):
                all_idx.add(i)
        return vessel_all[list(all_idx)]

    def vessel_filter_ray_close2(self, vessel_all: npt.NDArray, ray_cell_x_ind_all: npt.NDArray):
        """Filter the vessel that too close to the ray cells"""
        lx = len(self.params.x_vector)
        ly = len(self.params.y_vector)

        all_idx = set()
        for i, vessel in enumerate(vessel_all):
            diff = vessel[0] - ray_cell_x_ind_all
            if not np.any((diff >= -3) & (diff <= 4)) and vessel[0] <= lx - 3 and vessel[1] <= ly - 3:
                all_idx.add(i)
        return vessel_all[list(all_idx)]

    def vessel_extend(self, vessel_all: npt.NDArray):
        """Extend the vessel"""
        vessel_all_extend = np.empty((0, 3))
        for vessel in vessel_all:
            dist = vessel_all - vessel

            mark0 = np.where((dist[:, 0] <= 24) & (dist[:, 0] >= -8) & np.abs(dist[:, 1]) <= 8)[0]
            mark1 = np.where((dist[:, 0] <= 12) & (dist[:, 0] >= -6) & np.abs(dist[:, 1]) <= 6)[0]

            sign1 = np.random.choice([-1, 1])
            sign2 = np.random.choice([-1, 1])

            if len(mark0) > 1:
                vessel_all_extend = np.vstack((vessel_all_extend, vessel))
                possibility = np.random.rand(1)
                if len(mark1) <= 1:
                    if possibility < 0.2:
                        temp = [vessel[0] + 6 + sign1, vessel[1] + sign2 * 2]
                        vessel_all_extend = np.vstack((vessel_all_extend, temp))
                    else:
                        if possibility < 0.5:
                            temp = [vessel[0] + 6, vessel[1]]
                            vessel_all_extend = np.vstack((vessel_all_extend, temp))
            else:
                if vessel[0] + 12 < len(self.params.x_vector) and vessel[1] + 10 < len(self.params.y_vector):
                    temp0 = [vessel[0] + 5 + sign1, vessel[1]]
                    possibility = np.random.rand(1)
                    if possibility < 0.3:
                        temp = np.vstack((
                            temp0,
                            [temp0[0] + 5, temp0[1] + 2 * sign2]
                        ))
                    else:
                        temp = np.vstack((
                            temp0,
                            [temp0[0] + 5 + sign2, temp0[1]]
                        ))
                    vessel_all_extend = np.vstack((vessel_all_extend, vessel, temp))
                else:
                    vessel_all_extend = np.vstack((vessel_all_extend, vessel))
        return vessel_all_extend

    def fiber_filter_in_vessel(self, vessel_all: npt.NDArray):
        """This function is used to remove the fibers in the vessels"""
        lx = len(self.params.x_vector)

        indx_skip_all = np.empty((0, 6))
        indx_vessel_cen = []
        indx_vessel = np.empty((0, 6))
        # TODO: use np.ravel_multi_index here to improve performance (is it even needed if we keep grid non-flattened?)
        for i, vessel in enumerate(vessel_all):
            # skip all the cells inside the vessel
            vx, vy = vessel
            indx_skip = [
                (vx - 1) * lx + (vy - 2),
                (vx + 1) * lx + (vy - 2),
                (vx - 2) * lx + vy,
                (vx + 2) * lx + vy,
                (vx - 1) * lx + (vy + 2),
                (vx + 1) * lx + (vy + 2)
            ]
            # The six points used to fit the vessel
            indx_vessel = np.vstack((indx_vessel, [
                (vx - 3) * lx + (vy - 1),
                (vx - 3) * lx + (vy + 1),
                vx * lx + vy - 3,
                vx * lx + vy + 3,
                (vx + 3) * lx + (vy - 1),
                (vx + 3) * lx + (vy + 1)
            ]))

            indx_skip_all = np.vstack((indx_skip_all, indx_skip))
            indx_vessel_cen.append(vx * lx + vy)

        return indx_skip_all, indx_vessel, indx_vessel_cen

    def distrbute_ray_cells(self, ray_cell_x_ind_all: npt.NDArray):
        """Distribute the ray cells across the volume"""
        x_ind = []
        width = []
        keep = []
        x_ind_all_update = []
        if self.params.is_exist_ray_cell:
            ray_cell_num = self.params.ray_cell_num
            ray_cell_num_std = self.params.ray_cell_num_std
            ray_height = self.params.ray_height

            m = self.params.size_im_enlarge[2] / self.params.ray_cell_num / self.params.ray_height + 6
            for i, idx in enumerate(ray_cell_x_ind_all):
                app = [0]
                ray_cell_space = np.round(16 * np.random.rand(np.ceil(m)) + 6)
                rnd = np,round(-30 * np.random.rand())
                for rs in ray_cell_space:
                    ray_idx = [idx, idx + 1]
                    group = max(5, min(25,
                        np.round(np.random.randn() * ray_cell_num_std + ray_cell_num)
                    ))
                    app = app[-1] + (np.arange(group + 1) + rs + rnd) * ray_height
                    rnd = 0

                    if app[0] <= self.params.size_im_enlarge[2] - 150 and app[-1] >= 150:
                        x_ind.append(ray_idx)
                        x_ind_all_update.append(idx)
                        width.append(np.round(app))
                        keep.append(i)

        return (
            np.array(x_ind, dtype=int),
            np.array(width, dtype=float),
            np.array(keep, dtype=int),
            np.array(x_ind_all_update, dtype=int)
        )

    def get_fibers(self, x_ind: npt.NDArray, keep: npt.NDArray) -> npt.NDArray:
        """Generate small fibers."""
        return np.concatenate((
            x_ind[keep],
            x_ind[keep] + 1
        ))


    def generate(self):
        """Generate ray cells"""
        np.random.seed(self.params.random_seed)
        # ray_cell_length = self.params.ray_cell_length
        # ray_cell_variance = self.params.ray_cell_variance
        # ray_height = self.params.ray_height

        # size_im_enlarge = self.params.size_im_enlarge
        # x_grid = self.params.x_grid
        # cell_end_thick = self.params.cell_end_thick
        # cell_thick = self.params.cell_wall_thick

        # x_vector = self.params.x_vector
        # y_vector = self.params.y_vector

        # vessel_count = self.params.vessel_count

        x_grid_all, y_grid_all, thickness_all = self.get_grid_all()


        ray_cell_x_ind_all = self.get_ray_cell_indexes()
        vessel_all = self.get_vessels_all(ray_cell_x_ind_all)

        indx_skip_all, indx_vessel, indx_vessel_cen = self.fiber_filter_in_vessel(vessel_all)

        vol_img_ref = np.zeros(self.params.size_im_enlarge)

        ray_cell_x_ind, ray_cell_width, keep_ray_cell, ray_cell_x_ind_all_update = self.distrbute_ray_cells(ray_cell_x_ind_all)

        fibers = self.get_fibers(ray_cell_x_ind, keep_ray_cell)

    def ray_generate(self, vol_img_ref_final, ray_cell_x_ind: int, ray_cell_width: float):
        raise NotImplementedError('Function not implemented yet')
