try:
    from . import rbbox_geo_cuda
except:
    rbbox_geo_cuda = None
    print('no rbbox_geo_cuda package in build')


def rbbox_iou_iof(rb1, rb2, vec=False, iof=False):
    if vec:
        return rbbox_geo_cuda.vec_iou_iof(rb1, rb2, iof)
    else:
        return rbbox_geo_cuda.mat_iou_iof(rb1, rb2, iof)
