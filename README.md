# MMsAGN
Masked Graph Structure and Multi-scale Attention-based Graph Network for Vehicle Re-Identification

## Train
### Training on VeRi-776 dataset：
```
python tools/train_net.py --config-file ./configs/VeRi/sbs_R50-ibn.yml
```

### Training on VehicleID dataset：
```
python tools/train_net.py --config-file ./configs/VehicleID/Base-bagtricks.yml
```

### Training with four GPUs：

```
python3 tools/train_net.py --config-file ./configs/VeRi/sbs_R50-ibn.yml --num-gpus 4
```

## Test:

```
python tools/train_net.py --config-file ./configs/VeRi/sbs_R50-ibn.yml --eval-only MODEL.WEIGHTS ./logs/veri776/model_best.pth
```
