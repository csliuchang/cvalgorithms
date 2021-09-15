# RotateDetection 


## Performance

### MSRA-TD500
| Model |    Backbone    |  image size  | precision    |    recall    |    mAP  | GPU | Image/GPU | FPS | Loss| lr schd | Data Augmentation | Configs |       
|:------------:|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|     
| [RRetinaNet](https://arxiv.org/pdf/1707.06484.pdf)| yolov5s | 448x448 | 0.6647 | 0.7100 | 0.4956 | **1X** GeForce RTX 1660 Ti | 2 | 52 | FocalLoss GWDloss | 5e-5 1x | No | [dla_resnet18.json](./configs/rretinanet/models/rretinanet_yolov5_backbone.json) |
| [RRetinaNet](https://arxiv.org/pdf/1707.06484.pdf)| yolov5s | 800x800 | 0.8891 | 0.7876 | 0.7077 | **1X** GeForce RTX 1660 Ti | 2 | 48 | FocalLoss GWDloss | 1e-3 1x | No | [dla_resnet18.json](./configs/rretinanet/models/rretinanet_yolov5_backbone.json) |
| [RFCOS](https://arxiv.org/abs/1904.01355)| resnet18 | 800x800 | 0.8023 | 0.6498 | 0.5386 | **1X** GeForce RTX 1660 Ti | 2 | 40 | FocalLoss GWDloss CELoss | 1e-3 1x | No | [dla_resnet18.json](./configs/rretinanet/models/rretinanet_yolov5_backbone.json) |











