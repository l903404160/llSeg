# 2020年9月21日

## updates
1. 更新 Segmentation Head 的返回方式，
之前用很多if-else判断太不好看了，
现在使用inference函数统一进行预测，
在forward里面省去了判断是否要返回预测的过程
2. 这个更新还有待完善，目前更新了一些常用的Head，还有一些遗留Head没有更新，包括EMA，PC
3. 使用时有BUG及时调整，要做出更改就拉分支然后提交