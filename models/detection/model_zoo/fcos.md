# Model Name : FCOS: Fully Convolutional One-Stage Object Detection

## Backbone
- [x] ResNet 50

## Head
- [x] Deformable Conv wrapper
- [x] Forward features
- [x] Forward bbox
- [x] Inference
- [x] Losses
- [x] FCOS Head

## Performance 
- [x] R-50-FPN FCOS COCO_2017 38.6%
- [ ] R-50-FPN FCOS COCO_2017 Conv-2 in-progress
- [ ] R-50-FPN FCOS COCO_2017 Conv-6 in-progress

## Improvements
- [ ] Fuse a feature for designing another regression head.
- [ ] DARTS search a new FPN and new feature fusion processor.