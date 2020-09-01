# 语义分割数据集任务 - 1

## 1. 熟悉 dataset 程序，将ADE20K的数据集添加到 regisiter 里。
要求：
1. 在自己新分支下添加文件 `datasets/segmentation/ade20k.py`，并测试是否能够正确读取到数据`List`

> 参考：
> + `datasets`
>   + `segmentation`
>       + `builder.py`
>       + `cityscapes.py`
>       + `regisiter.py`
>       + `seg_builtin.py`
>       + `seg_builtin_meta.py`
>   + `metacatalog`
>       + `catalog.py`
>

## 2. 测试是否在 regisiter 中注册成功。
> 运行 `datasets/test/regisiter_test.py`，查看自己写的数据集是否在列表之中。

# 语义分割数据集任务 - 2

## 1. 熟悉数据是如何进行读取和处理（包括一些transform）

要求:
1. 熟悉数据读取和处理
2. 在`datasets/test`下新建一个文件，测试ADE20K数据读取是否正常

> 参考：
> + `datasets`
>   + `segmentation`
>       + `builder.py`
>       + `cityscapes.py`
>       + `seg_mapper.py`
>   + `common`
>       + all
>   + `transforms`
>       + all
