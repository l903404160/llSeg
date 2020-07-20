# CODING FOR CORNERNET --

## Paper Reading

[CornerNet: Detecting Objects as Paired Keypoints](https://arxiv.org/pdf/1808.01244.pdf)

1. apporaching object detection to pair of keypoints(top-left and bottom-right) detection task.

> DeNet / Point Linking Network (PLN)

## Coding Task

### Data Loading
- [x] COCO
- [x] AnchorFree Mapper
- [x] transforms
    - [x] Read Image
    - [x] Resize from scales (scales)
    - [x] Keep the short edge
    - [x] Crop image (511, 511)
    - [x] Random Flip


### Backbone

- [x] HourGlass
- [ ] <del>Pretrian Weight Loading</del> ps: In the CornerNet paper, the author say they do not use any pretrained model.
- [x] Adjust to Detectron2 API

### Detection Head
- [x] CornerNet Head Prediction
- [x] Transpose and Gather feature
- [x] CornerNet Head Loss
- [x] CornerNet Head Inference

### Base Model 
- [x] Normalization (Placed in Model Class)
- [x] Draw Gaussian
- [x] Computation Regr

### Performance validation
- [x] Loss check
- [x] Model check

### Inference
- [x] NMS
- [x] Convert to Instances for adjusting the Detectron2 API
- [x] Single Image Testing
- [x] Other Problem
    - [x] Move some local variables to config file
    - [x] Another single image testing, because of the bad predictions

## Conclusion
- [ ] Performance Evaluation