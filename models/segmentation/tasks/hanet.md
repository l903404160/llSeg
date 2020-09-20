# HANet
## Implementation Tsak

- [x] implement the aspp module
- [x] implement the HANet Conv
- [x] implemment the HANet head
- [x] register the head into segmentation head registry
- [x] evaluate the final performance
- [ ] try to make improvement

需要完成的文件列表如下：
1. `models/segmentation/heads/hanet_head.py`
1. `models/segmentation/segmods/hanetmods.py`

各位完成各自的任务之后，我将会进行性能测试，然后看看是不是存在问题，
截止`明天下午2点之前`大家要完成各自的程序任务.

## Performance Evaluation

|Model| Backbone | Stride | Datasets | MIoU|
|-|-|-|-|-|
|HANet| ResNet-101|8| CityScapes |81.07%|