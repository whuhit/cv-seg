# FCN

## 关于FCN

这是FCN的pytorch复现代码。用的不是作者文章中的PASCAL VOC数据集，而是一个更小的道路场景数据集Camvid。作者文中说到需要训练700轮才能达到一个很好的效果。但是Camvid数据集比较小，我训练不到200轮就已经严重过拟合了，训练集mIOU可接近90%，在验证集上最好的mIoU为59.94%，而在测试集上mIoU则在52%左右。



## 运行方法

### 训练

```python
python train.py
```

### 测试

1、下载预训练模型链接: https://pan.baidu.com/s/1DwxUNK3ZM8TgpFay1PzclA  密码: b909，并放入相应文件夹。

```python
python test.py
```



## 优化空间





