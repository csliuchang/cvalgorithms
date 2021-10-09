# Modified by Chang Liu
# Contact: liuchang@deepsight.ai


import torch
from .comm import fp16_clamp
from torch import Tensor
import math
from typing import Tuple
import cv2
import numpy as np
import math

pi = math.pi

_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def cxcywh_xyxy(bbox):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    return np.array([x - w/2, y - h/2, x + w/2, y + h/2], dtype=np.float32)

def xyxy_cxcywh(bbox):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def rbox_2_quad(rboxes, mode='xyxya'):
    if len(rboxes.shape) == 1:
        rboxes = rboxes[np.newaxis, :]
    if rboxes.shape[0] == 0:
        return rboxes
    quads = np.zeros((rboxes.shape[0], 8), dtype=np.float32)
    for i, rbox in enumerate(rboxes):
        if len(rbox != 0):
            if mode == 'xyxya':
                w = rbox[2] - rbox[0]
                h = rbox[3] - rbox[1]
                x = rbox[0] + 0.5 * w
                y = rbox[1] + 0.5 * h
                theta = rbox[4]
            elif mode == 'xywha':
                x = rbox[0]
                y = rbox[1]
                w = rbox[2]
                h = rbox[3]
                theta = rbox[4]
            quads[i, :] = cv2.boxPoints(((float(x), float(y)), (float(w), float(h)), float(theta))).reshape((1, 8))

    return quads


def xy2wh(boxes):
    """
    :param boxes: (xmin, ymin, xmax, ymax) (n, 4)
    :return: out_boxes: (x_ctr, y_ctr, w, h) (n, 4)
    """
    if torch.is_tensor(boxes):
        out_boxes = boxes.clone()
    else:
        out_boxes = boxes.copy()
    out_boxes[:, 2] = boxes[:, 2] - boxes[:, 0] + 1.0
    out_boxes[:, 3] = boxes[:, 3] - boxes[:, 1] + 1.0
    out_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5
    out_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5

    return out_boxes


def ex_box_jaccard(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    inter_x1 = np.maximum(np.min(a[:, 0]), np.min(b[:, 0]))
    inter_x2 = np.minimum(np.max(a[:, 0]), np.max(b[:, 0]))
    inter_y1 = np.maximum(np.min(a[:, 1]), np.min(b[:, 1]))
    inter_y2 = np.minimum(np.max(a[:, 1]), np.max(b[:, 1]))
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.
    x1 = np.minimum(np.min(a[:, 0]), np.min(b[:, 0]))
    x2 = np.maximum(np.max(a[:, 0]), np.max(b[:, 0]))
    y1 = np.minimum(np.min(a[:, 1]), np.min(b[:, 1]))
    y2 = np.maximum(np.max(a[:, 1]), np.max(b[:, 1]))
    mask_w = np.int(np.ceil(x2 - x1))
    mask_h = np.int(np.ceil(y2 - y1))
    mask_a = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    mask_b = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    a[:, 0] -= x1
    a[:, 1] -= y1
    b[:, 0] -= x1
    b[:, 1] -= y1
    mask_a = cv2.fillPoly(mask_a, pts=np.asarray([a], 'int32'), color=1)
    mask_b = cv2.fillPoly(mask_b, pts=np.asarray([b], 'int32'), color=1)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    iou = float(inter) / (float(union) + 1e-12)
    # cv2.imshow('img1', np.uint8(mask_a*255))
    # cv2.imshow('img2', np.uint8(mask_b*255))
    # k = cv2.waitKey(0)
    # if k==ord('q'):
    #     cv2.destroyAllWindows()
    #     exit()
    return iou


def reorder_pts(tt, rr, bb, ll):
    pts = np.asarray([tt, rr, bb, ll], np.float32)
    l_ind = np.argmin(pts[:, 0])
    r_ind = np.argmax(pts[:, 0])
    t_ind = np.argmin(pts[:, 1])
    b_ind = np.argmax(pts[:, 1])
    tt_new = pts[t_ind, :]
    rr_new = pts[r_ind, :]
    bb_new = pts[b_ind, :]
    ll_new = pts[l_ind, :]
    return tt_new, rr_new, bb_new, ll_new


def cal_bbox_wh(pts_4):
    x1 = np.min(pts_4[:, 0])
    x2 = np.max(pts_4[:, 0])
    y1 = np.min(pts_4[:, 1])
    y2 = np.max(pts_4[:, 1])
    return x2 - x1, y2 - y1


def cal_bbox_pts(pts_4):
    x1 = np.min(pts_4[:, 0])
    x2 = np.max(pts_4[:, 0])
    y1 = np.min(pts_4[:, 1])
    y2 = np.max(pts_4[:, 1])
    bl = [x1, y2]
    tl = [x1, y1]
    tr = [x2, y1]
    br = [x2, y2]
    return np.asarray([bl, tl, tr, br], np.float32)


def quad_2_rbox(quads, mode='xywha'):
    # http://fromwiz.com/share/s/34GeEW1RFx7x2iIM0z1ZXVvc2yLl5t2fTkEg2ZVhJR2n50xg
    # long side method  (x, y ,w , h, a)
    if len(quads.shape) == 1:
        quads = quads[np.newaxis, :]
    rboxes = np.zeros((quads.shape[0], 5), dtype=np.float32)
    for i, quad in enumerate(quads):
        rbox = cv2.minAreaRect(quad.reshape([4, 2]))
        x, y, w, h, t = rbox[0][0], rbox[0][1], rbox[1][0], rbox[1][1], rbox[2]
        x = np.clip(x, 0, 448)
        y = np.clip(y, 0, 448)
        w = np.clip(w, 0, 448)
        h = np.clip(h, 0, 448)
        if np.abs(t) < 45.0:
            rboxes[i, :] = np.array([x, y, w, h, t])
        elif np.abs(t) > 45.0:
            rboxes[i, :] = np.array([x, y, h, w, 90.0 + t])
        else:
            if w > h:
                rboxes[i, :] = np.array([x, y, w, h, -45.0])
            else:
                rboxes[i, :] = np.array([x, y, h, w, 45])
    # (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    if mode == 'xyxya':
        rboxes[:, 0:2] = rboxes[:, 0:2] - rboxes[:, 2:4] * 0.5
        rboxes[:, 2:4] = rboxes[:, 0:2] + rboxes[:, 2:4]
    rboxes[:, 0:4] = rboxes[:, 0:4].astype(np.float32)
    return rboxes


def points2rdets(bboxes):
    # add bbox is [] return
    if not bboxes.any():
        return np.array([] * 5, dtype=np.float32).reshape(-1, 5)
    rbboxes = []
    for bbox in bboxes:
        bboxps = np.array(bbox).reshape(
            (4, 2)).astype(np.float32)

        rbbox = cv2.minAreaRect(bboxps)
        x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
        if w == 0 or h == 0:
            continue
        while not 0 > a >= -90:
            if a >= 0:
                a -= 90
                w, h = h, w
            else:
                a += 90
                w, h = h, w
        a = a / 180 * np.pi
        assert 0 > a >= -np.pi / 2
        rbbox = np.array([x, y, w, h, a], dtype=np.float32)
        rbboxes.append(rbbox)
    return np.stack(rbboxes, axis=0)


def ranchor_inside_flags(flat_ranchors,
                         valid_flags,
                         img_shape,
                         allowed_border=0):
    img_h, img_w = img_shape
    cx, cy = (flat_ranchors[:, i] for i in range(2))
    inside_flags = valid_flags & \
                   (cx >= -allowed_border) & \
                   (cy >= -allowed_border) & \
                   (cx < img_w + allowed_border) & \
                   (cy < img_h + allowed_border)

    return inside_flags


def rdets2points(rbboxes):
    """Convert detection results to a list of numpy arrays.

    Args:
        rbboxes (np.ndarray): shape (n, 6), xywhap encoded

    Returns:
        rbboxes (np.ndarray): shape (n, 9), x1y1x2y2x3y3x4y4p
    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    prob = rbboxes[:, 5]
    cls = rbboxes[:, 6]
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, prob, cls], axis=-1)


def rbbox2circumhbbox(rbboxes):
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    hbbox_w = cosa * w - sina * h
    hbbox_h = - sina * w + cosa * h
    # -pi/2 < a <= 0, so cos(a)>0, sin(a)<0
    hbboxes = rbboxes.clone().detach()
    hbboxes[:, 2::5] = hbbox_h
    hbboxes[:, 3::5] = hbbox_w
    hbboxes[:, 4::5] = -np.pi / 2
    return hbboxes


def rdets2points_tensor(rbboxes):
    """
    A tensor version of rdets2points
    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    prob = rbboxes[:, 5]
    cls = rbboxes[:, 6]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, prob, cls], dim=-1)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def regular_theta(theta, mode='180', start=-pi / 2):
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start


def mintheta_obb(obboxes):
    x, y, w, h, theta = obboxes.unbind(dim=-1)
    theta1 = regular_theta(theta)
    theta2 = regular_theta(theta + pi / 2)
    abs_theta1 = torch.abs(theta1)
    abs_theta2 = torch.abs(theta2)

    w_regular = torch.where(abs_theta1 < abs_theta2, w, h)
    h_regular = torch.where(abs_theta1 < abs_theta2, h, w)
    theta_regular = torch.where(abs_theta1 < abs_theta2, theta1, theta2)

    obboxes = torch.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)
    return obboxes


def distance2obb(points, distance, max_shape=None):
    distance, theta = distance.split([4, 1], dim=1)

    Cos, Sin = torch.cos(theta), torch.sin(theta)
    Matrix = torch.cat([Cos, Sin, -Sin, Cos], dim=1).reshape(-1, 2, 2)

    wh = distance[:, :2] + distance[:, 2:]
    offset_t = (distance[:, 2:] - distance[:, :2]) / 2
    offset_t = offset_t.unsqueeze(2)
    offset = torch.bmm(Matrix, offset_t).squeeze(2)
    ctr = points + offset

    obbs = torch.cat([ctr, wh, theta], dim=1)
    return regular_obb(obbs)


def regular_obb(obboxes):
    x, y, w, h, theta = obboxes.unbind(dim=-1)
    w_regular = torch.where(w > h, w, h)
    h_regular = torch.where(w > h, h, w)
    theta_regular = torch.where(w > h, theta, theta + pi / 2)
    theta_regular = regular_theta(theta_regular)
    return torch.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)
