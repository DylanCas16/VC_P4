# VC_P4

## Dependencias (notas, pendiente de ordenar)

### Entorno `VC_P4` (conda, Python 3.9.25) — YOLO

- torch 2.8.0+cu126
- torchvision 0.23.0+cu126
- ultralytics 8.4.200
- ultralytics-thop 2.2.4
- numpy 2.0.2
- opencv-python 5.0.0.93
- opencv-contrib-python 4.10.0.84
- matplotlib 3.9.4
- pandas 2.3.3
- scipy 1.13.1
- pillow 11.3.0
- PyYAML 6.0.2
- tqdm 4.70.1
- jupyter 1.1.1, jupyterlab 4.5.11, notebook 7.5.8, ipykernel 6.31.0

GPU: NVIDIA GeForce RTX 4050 Laptop (driver 617.42), CUDA 12.6.

### PaddleOCR instalado hasta ahora en `VC_P4` (esto provoca el conflicto de cuDNN con torch)

- paddlepaddle-gpu 3.3.1
- paddleocr 3.7.0
- paddlex 3.7.2
- nvidia-cublas-cu12 12.6.4.1
- nvidia-cuda-runtime-cu12 12.6.77
- nvidia-cudnn-cu12 9.5.1.17
- nvidia-cufft-cu12 11.3.0.4
- nvidia-curand-cu12 10.3.7.77
- nvidia-cusolver-cu12 11.7.1.2
- nvidia-cusparse-cu12 12.5.4.2
- nvidia-nvjitlink-cu12 12.9.86
- nvidia-ml-py 13.615.71

Instalados a mano para el entrenamiento de PaddleOCR (con `--no-deps`):

- albumentations 2.0.8
- albucore 0.0.24 (la 0.0.33 es incompatible con albumentations 2.0.8)
- scikit-image 0.24.0
- lmdb 3.0.0
- imageio 2.37.2
- lazy-loader 0.6
- Cython 3.3.1
- eval_type_backport 0.4.0

Ya presentes y usados por PaddleOCR: shapely 2.0.7, pyclipper 1.3.0.post6, RapidFuzz 3.13.0, networkx 3.2.1, tifffile 2024.8.30.

### Problema encontrado

`PaddleOCR/tools/train.py` importa `paddle` primero y luego `albumentations` importa `torch`.
Las dos librerías cargan su propia `cudnn_cnn64_9.dll` y falla con `WinError 127`.
Solución elegida: entorno aparte `paddle`, sin torch (ver abajo).

### Entorno `paddle` (conda, Python 3.9) — PENDIENTE DE CREAR

```bash
conda create -n VC_paddle python=3.9 -y
conda activate VC_paddle
python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install paddleocr==3.7.0
python -m pip install -r PaddleOCR/requirements.txt
python -m pip install ipykernel matplotlib
```

`PaddleOCR/requirements.txt` incluye: shapely, scikit-image, pyclipper, lmdb, tqdm, numpy, rapidfuzz, opencv-python, opencv-contrib-python, cython, Pillow, pyyaml, requests, albumentations, albucore, packaging.

No instalar `torch` ni `ultralytics` en este entorno.
