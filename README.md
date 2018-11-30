# YOLO

### BaseNet Train
basenet有两种，一是论文复现，二是卷积层后添加了BN层的。
```python
python basenet_train.py --image_size=224 --batch_size=32 --num_classes=1000 --dataset="imagenet_224" --net_chose="base_net_bn" --train_data_path="" --val_data_path=""
```
```python
python basenet_train.py --batch_size=16 --num_classes=17 --dataset="flowers17_448" --net_chose="base_net_bn" --train_data_path="" --val_data_path=""
```

### Yolo v1 Train
训练集选用VOC 21分类数据集。
```python

```