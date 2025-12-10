# PyTorch SuperPoint Baseline

PyTorch реализация статьи ["SuperPoint: Self-Supervised Interest Point Detection and Description"](https://arxiv.org/abs/1712.07629). Основано на TensorFlow реализации [rpautrat/SuperPoint](https://github.com/rpautrat/SuperPoint).

## Результаты на HPatches
| Task | Homography estimation | | | Detector metric | | Descriptor metric | |
|------|-----------------------|-|-|-----------------|-|-------------------|-|
| | Epsilon = 1 | 3 | 5 | Repeatability | MLE | NN mAP | Matching Score |
| Sift (subpixel accuracy) | 0.63 | 0.76 | 0.79 | 0.51 | 1.16 | 0.70 | 0.27 |
| superpoint_coco_heat2_0_170k_hpatches_sub | 0.46 | 0.75 | 0.81 | 0.63 | 1.07 | 0.78 | 0.42 |
| superpoint_coco(my_model)| 0.42 | 0.75 | 0.81 | 0.59 | 1.05 | 0.866 | 0.516 |

## Установка
### Требования
- Python 3.6+
- PyTorch ≥ 1.1
- CUDA (≥ 10)

```
conda create --name superpoint python=3.12
conda activate superpoint
pip install -r requirements.txt
pip install -r requirements_torch.txt
```

### Настройка путей
Пути к данным и логам задаются в `setting.py`.

## Данные
Скачайте датасеты в `$DATA_DIR`:
```
datasets/
├── COCO
│   ├── train2014
│   └── val2014
├── HPatches
└── synthetic_shapes  # создаётся автоматически
```
- **COCO 2014**: [ссылка](http://cocodataset.org/#download)
- **HPatches**: [ссылка](https://www.kaggle.com/datasets/javidtheimmortal/hpatches)

## Обучение
### 1. Обучение MagicPoint на синтетических данных
```
python train4.py train_base configs/magicpoint_shapes_pair.yaml magicpoint_synth --eval
```
### 2. Экспорт детекций с COCO(Homography Adaptation)
**COCO (train):**
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```
**COCO (val):**
- Измените `export_folder` на `val` в `magicpoint_coco_export.yaml`
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```
### 3. Обучение SuperPoint на COCO
```
python train4.py train_joint configs/superpoint_coco_train_heatmap.yaml superpoint_coco --eval --debug
```
### 4. Экспорт и оценка на HPatches
```
# Экспорт
python export.py export_descriptor configs/magicpoint_repeatability_heatmap.yaml superpoint_hpatches_test
# Оценка
python evaluation.py logs/superpoint_hpatches_test/predictions --repeatibility --outputImg --homography --plotMatching
```
### 5. Сравнение с SIFT
```
# Экспорт SIFT
python export_classical.py export_descriptor configs/classical_descriptors.yaml sift_test --correspondence
# Оценка
python evaluation.py logs/sift_test/predictions --sift --repeatibility --homography
```

## Предобученные модели
### Лучшая модель из оригинального репозитория
`logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar`
### Моя модель (обученная на COCO)
`logs/superpoint_coco/superPointNet_200000_checkpoint.pth.tar`
