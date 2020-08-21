# llSeg

This repo is constructed for reproducing many interesting DL algorithms. Now it is still improving and I will check the model performance as soon as possible.

# Status: Constructing

## Semantic Segmentation

### MODEL Statistics

- [x] DeeplabV3
- [ ] PSPNet
- [x] EMANet
- [x] ANN

### Performance Statistics

|Model| Backbone | Stride | Datasets | MIoU|
|-|-|-|-|-|
|BaseLine| ResNet-101|16| CityScapes |75.1%|

## Object Detection

- [x] Faster RCNN
- [x] Cascade RCNN
- [x] RetinaNet
- [x] FCOS
- [ ] BorderDet

### Performance Statistics

|Model| Backbone | Stride | Datasets | mAP|
|-|-|-|-|-|
|FCOS| ResNet-50-FPN| - | COCO-2017 Val |38.7%|

> Tips: It is not easy to use in just one line command. May you need to be clear about `detectron2`.
> More concrete usage will be updated after I completed the FCOS code.

>Refs:
> 1. [detectron2](https://github.com/facebookresearch/detectron2)
> 2. [fvcore](https://github.com/facebookresearch/fvcore)