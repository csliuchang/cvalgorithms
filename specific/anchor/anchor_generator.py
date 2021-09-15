import numpy as np
import torch
from torch.nn.modules.utils import _pair

from .builder import ANCHOR_GENERATORS
import torch.nn as nn
import collections
import math
from typing import List


@ANCHOR_GENERATORS.register_module()
class AnchorGenerator(object):
    """Standard anchor generator for 2D anchor-based detectors.

    Parameters
    ----------
    strides : list[int] | list[tuple[int, int]]
        Strides of anchors
        in multiple feature levels in order (w, h).
    ratios : list[float]
        The list of ratios between the height and width
        of anchors in a single level.
    scales : list[int] | None
        Anchor scales for anchors in a single level.
        It cannot be set at the same time if `octave_base_scale` and
        `scales_per_octave` are set.
    base_sizes : list[int] | None
        The basic sizes
        of anchors in multiple levels.
        If None is given, strides will be used as base_sizes.
        (If strides are non square, the shortest stride is taken.)
    scale_major : bool
        Whether to multiply scales first when generating
        _base anchors. If true, the anchors in the same row will have the
        same scales. By default it is True in V2.0
    octave_base_scale : int
        The _base scale of octave.
    scales_per_octave : int
        Number of scales for each octave.
        `octave_base_scale` and `scales_per_octave` are usually used in
        retinanet and the `scales` should be None when they are set.
    centers : list[tuple[float, float]] | None
        The centers of the anchor
        relative to the feature grid center in multiple feature levels.
        By default it is set to be None and not used. If a list of tuple of
        float is given, they will be used to shift the centers of anchors.
    center_offset : float
        The offset of center in proportion to anchors'
        width and height. By default it is 0.

    Examples
    --------
    >>> from specific import AnchorGenerator
    >>> self = AnchorGenerator([16], [1.], [1.], [9])
    >>> all_anchors = self.grid_anchors([(2, 2)], device='cpu')
    >>> print(all_anchors)
    [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
            [11.5000, -4.5000, 20.5000,  4.5000],
            [-4.5000, 11.5000,  4.5000, 20.5000],
            [11.5000, 11.5000, 20.5000, 20.5000]])]
    >>> self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
    >>> all_anchors = self.grid_anchors([(2, 2), (1, 1)], device='cpu')
    >>> print(all_anchors)
    [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
            [11.5000, -4.5000, 20.5000,  4.5000],
            [-4.5000, 11.5000,  4.5000, 20.5000],
            [11.5000, 11.5000, 20.5000, 20.5000]]),
    tensor([[-9., -9., 9., 9.]])]
    """

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 auto_anchors=None,
                 center_offset=0.):
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                                    f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        # assert len(self.base_sizes) == len(self.strides), \
        #     'The number of strides should be the same as _base sizes, got ' \
        #     f'{self.strides} and {self.base_sizes}'

        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.auto_anchors = auto_anchors
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int]: total number of _base anchors in a feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate _base anchors.

        Returns
        -------
        list : torch.Tensor
            Base anchors of a feature grid in multiple feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            if self.auto_anchors:
                multi_level_base_anchors.append(
                    self.gen_single_level_auto_anchors(
                        base_size,
                        center=center
                    )
                )

            else:
                multi_level_base_anchors.append(
                    self.gen_single_level_base_anchors(
                        base_size,
                        scales=self.scales,
                        ratios=self.ratios,
                        center=center))
        return multi_level_base_anchors

    def gen_single_level_auto_anchors(self,
                                      base_size,
                                      center=None):
        """
        Generate _base anchors by auto anchor
        base_size : (int | float, int | float)
                    Basic size of an anchor.
        """
        w = torch.tensor(base_size[:, 0])
        h = torch.tensor(base_size[:, 1])
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        x_center += torch.zeros_like(w)
        y_center += torch.zeros_like(h)

        base_anchors = torch.stack(
            [x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w,
             y_center + 0.5 * h], dim=-1
        )

        return base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate _base anchors of a single level.

        Parameters
        ----------
        base_size : int | float
            Basic size of an anchor.
        scales : torch.Tensor
            Scales of the anchor.
        ratios : torch.Tensor
            The ratio between between the height
            and width of anchors in a single level.
        center : tuple[float], optional
            The center of the _base anchor
            related to a single feature grid. Defaults to None.

        Returns
        -------
        torch.Tensor
            Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.

        Parameters
        ----------
        x : torch.Tensor
            Grids of x dimension.
        y : torch.Tensor
            Grids of y dimension.
        row_major : bool, optional
            Whether to return y grids first.
            Defaults to True.

        Returns
        -------
        tuple[torch.Tensor]
            The mesh grids of x and y.
        """
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_sizes, device='cuda'):
        """Generate grid anchors in multiple feature levels.

        Parameters
        ----------
        featmap_sizes : list[tuple]
            List of feature map sizes in
            multiple feature levels.
        device : str
            Device where the anchors will be put on.

        Returns
        -------
        list[torch.Tensor]
            Anchors in multiple feature levels.
            The sizes of each tensor should be [N, 4], where
            N = width * height * num_base_anchors, width and height
            are the sizes of the corresponding feature level,
            num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i].to(device),
                featmap_sizes[i],
                self.strides[i],
                device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        """Generate grid anchors of a single level.

        Notes
        -----
        This function is usually called by method ``self.grid_anchors``.

        Parameters
        ----------
        base_anchors : torch.Tensor
            The _base anchors of a feature grid.
        featmap_size : tuple[int]
            Size of the feature maps.
        stride : tuple[int], optional
            Stride of the feature map in order
            (w, h). Defaults to (16, 16).
        device : str, optional
            Device the tensor will be put on.
            Defaults to 'cuda'.

        Returns
        -------
        torch.Tensor
            Anchors in the overall feature maps.
        """
        feat_h, feat_w = featmap_size
        feat_h = int(feat_h)
        feat_w = int(feat_w)
        shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        shift_y = torch.arange(0, feat_h, device=device) * stride[1]

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        return all_anchors

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of anchors in multiple feature levels.

        Parameters
        ----------
        featmap_sizes : list(tuple)
            List of feature map sizes in
            multiple feature levels.
        pad_shape : tuple
            The padded shape of the image.
        device : str
            Device where the anchors will be put on.

        Returns
        -------
        list : torch.Tensor
            Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[0])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[1])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors,
                                 device='cuda'):
        """Generate the valid flags of anchor in a single feature map.

        Parameters
        ----------
        featmap_size : tuple[int]
            The size of feature maps.
        valid_size : tuple[int]
            The valid size of the feature maps.
        num_base_anchors : int
            The number of _base anchors.
        device : str, optional
            Device where the flags will be put on.
            Defaults to 'cuda'.

        Returns
        -------
        torch.Tensor
            The valid flags of each anchor in a single level feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}octave_base_scale='
        repr_str += f'{self.octave_base_scale},\n'
        repr_str += f'{indent_str}scales_per_octave='
        repr_str += f'{self.scales_per_octave},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}centers={self.centers},\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(i), buffer)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _create_grid_offsets(size: List[int], stride: int, offset: float, device: torch.device):
    grid_height, grid_width = size
    shifts_x = torch.arange(
        offset * stride, grid_width * stride, step=stride, dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        offset * stride, grid_height * stride, step=stride, dtype=torch.float32, device=device
    )
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)
    shifts_x = shifts_x.reshape(-1)
    shifts_y = shifts_y.reshape(-1)
    return shifts_x, shifts_y


def _broadcast_params(params, num_features, name):
    """
    If one size (or aspect ratio) is specified and there are multiple feature
    maps, we "broadcast" anchors of that single size (or aspect ratio)
    over all feature maps.

    If params is list[float], or list[list[float]] with len(params) == 1, repeat
    it num_features time.

    Returns:
        list[list[float]]: param for each feature
    """
    # assert isinstance(
    #     params, collections.abc.Sequence
    # ), f"{name} in anchor generator has to be a list! Got {params}."
    assert len(params), f"{name} in anchor generator cannot be empty!"
    if not isinstance(params[0], collections.abc.Sequence):  # params is list[float]
        return [params] * num_features
    if len(params) == 1:
        return list(params) * num_features
    assert len(params) == num_features, (
        f"Got {name} of length {len(params)} in anchor generator, "
        f"but the number of input features is {num_features}!"
    )
    return params


@ANCHOR_GENERATORS.register_module()
class DefaultAnchorGenerator(nn.Module):
    """
    Compute anchors in the standard ways described in
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".
    """

    box_dim: torch.jit.Final[int] = 4
    """
    the dimension of each anchor box.
    """

    def __init__(self, *, sizes, aspect_ratios, strides, offset=0.5):
        """
        This interface is experimental.

        Args:
            sizes (list[list[float]] or list[float]):
                If ``sizes`` is list[list[float]], ``sizes[i]`` is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If ``sizes`` is list[float], ``sizes`` is used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
            strides (list[int]): stride of each input feature.
            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
        """
        super().__init__()

        self.strides = strides
        self.num_features = len(self.strides)
        sizes = _broadcast_params(sizes, self.num_features, "sizes")
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

        self.offset = offset
        assert 0.0 <= self.offset < 1.0, self.offset

    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    @property
    @torch.jit.unused
    def num_cell_anchor(self):
        """
        Alias of 'num_anchor'.
        """
        return self.num_anchors

    @property
    @torch.jit.unused
    def num_anchor(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)

                In standard RPN retinanet, `num_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
        """
        anchors = []
        # buffers() not supported by torchscript. use named_buffers() instead
        buffers: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227
        anchors = []
        for size in sizes:
            # area = size ** 2.0
            # for aspect_ratio in aspect_ratios:
            # w = math.sqrt(area / aspect_ratio)
            # h = aspect_ratio * w
            w, h = size[0], size[1]
            x0, y0, x1, y1, = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(self, feature_sizes=[[28, 28]]):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[Boxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """

        grid_sizes = [feature_size for feature_size in feature_sizes]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return [x for x in anchors_over_all_feature_maps]
