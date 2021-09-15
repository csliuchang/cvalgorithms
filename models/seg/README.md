# segmentation 


## Performance

### Cityscapes500
| Model |    Backbone    |    Training data    |    Val data    |    mIOU  | GPU | Image/GPU | FPS | Loss| lr schd | Data Augmentation | Configs |       
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|     
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)| ResNet18 | Cityscapes train | Cityscapes val | 0.4957 | **1X** GeForce RTX 1660 Ti | 2 | 101 | CELoss | 2x | No | [dla_resnet18.json](./configs/seg/models/dla/dla_resnet18.json) |
| [polyp](https://arxiv.org/pdf/2108.06932.pdf)| yolov5s | Cityscapes train | Cityscapes val | 0.4644 | **1X** GeForce RTX 1660 Ti | 2 | 86 | OhemCELoss | 2x | No | [r3det_r50_fpn_2x_CustomizeImageSplit.py](./configs/r3det/r3det_r50_fpn_2x_CustomizeImageSplit.py) |
| [STDC](https://arxiv.org/abs/2104.13188)| ResNet18 | Cityscapes train | Cityscapes val | 0.4736 | **1X** GeForce RTX 1660 Ti | 2 | 78 | OhemCELoss | 2x | No | [stdc_resnet18.json](./configs/seg/models/stdc/stdc_resnet18.json) |
| [STDC<sup>813](https://arxiv.org/abs/2104.13188)| STDC813 | Cityscapes train | Cityscapes val | 0.4920 | **1X** GeForce RTX 1660 Ti | 2 | 112 | OhemCELoss | 2x | No | [stdc_stdcnet813.json](./configs/seg/models/stdc/stdc_stdcnet813.json) |
| [OCRNet](https://arxiv.org/pdf/1909.11065 )| HRNet18 | Cityscapes train | Cityscapes val | 0.5153 | **1X** GeForce RTX 1660 Ti | 2 | 13 | OhemCELoss | 2x | No | [ocrnet_hrnet18.json](./configs/seg/models/ocrnet/ocrnet_hrnet18.json) |

### HCI
| Model |    Backbone    |    Training data    |    Val data    |    mIOU  | GPU | Image/GPU | FPS | Loss| lr schd | Data Augmentation | Configs |       
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|     
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)| ResNet18 | HCI train | HCI val | 0.9495 | **1X** GeForce RTX 1660 Ti | 2 | 101 | CELoss  | 2x | No | [dla_resnet18.json](./configs/seg/models/dla/dla_resnet18.json) |
| [polyp](https://arxiv.org/pdf/2108.06932.pdf)| yolov5s | HCI train | HCI val | 0.9362 | **1X** GeForce RTX 1660 Ti | 2 | 86 | OhemCELoss | 2x | No | [r3det_r50_fpn_2x_CustomizeImageSplit.py](./configs/r3det/r3det_r50_fpn_2x_CustomizeImageSplit.py) |
| [STDC](https://arxiv.org/abs/2104.13188)| ResNet18 | HCI train | HCI val | 0.9348 | **1X** GeForce RTX 1660 Ti | 2 | 78 | OhemCELoss | 2x | No | [stdc_resnet18.json](./configs/seg/models/stdc/stdc_resnet18.json) |
| [STDC<sup>813](https://arxiv.org/abs/2104.13188)| STDC813 | HCI train | HCI val | 0.9308 | **1X** GeForce RTX 1660 Ti | 2 | 112 | OhemCELoss | 2x | No | [stdc_stdcnet813.json](./configs/seg/models/stdc/stdc_stdcnet813.json) |
| [OCRNet](https://arxiv.org/pdf/1909.11065 )| HRNet18 | HCI train | HCI val | 0.9439 | **1X** GeForce RTX 1660 Ti | 2 | 13 | OhemCELoss | 2x | No | [ocrnet_hrnet18.json](./configs/seg/models/ocrnet/ocrnet_hrnet18.json) |