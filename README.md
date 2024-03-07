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
You can use our trained weight files for testing. You can access the files through [this link](https://drive.google.com/drive/folders/1r3W7dDekqBfmKHSPsNyF8fQKhcqCbmrI?usp=sharing).

## Result

### VeRi-776

Performance (%) comparison on the VeRi-776 dataset.
| Method | mAP   | Rank-1 | Rank-5 |
|--------|-------|--------|--------|
| Baseline| 81.09 | 96.72  | 98.33  |
|MMsAGN| **83.57** | **97.44** | **99.05** |

### VehicleID

Performance (%) comparison on the VehicleID dataset.

| Method | Test800 Rank-1 | Test800 Rank-5 | Test1600 Rank-1 | Test1600 Rank-5 | Test2400 Rank-1 | Test2400 Rank-5 |
|--------|----------------|----------------|-----------------|-----------------|-----------------|-----------------|
| Baseline | 66.33 | 89.38 | 58.72 | 82.55 | 53.77 | 76.83 |
| MMsAGN | **87.41** | **98.17** | **83.88** | **96.36** | **81.30** | **94.22** |

