# VC_P4 - Reconocimiento y lector de matrículas

## Dataset utilizado

El dataset utilizado está formado por 2184 imágenes etiquetadas, las cuales contienen tanto la posición como los números de matrícula, que serán útiles para la segunda parte de la práctica. El dataset se puede descargar en el siguiente drive: [dataset](https://drive.google.com/file/d/1IETJJqb2CJnpKCwmm6MfeaVMvr0hM4vM/view?usp=drive_link)

Para importar el dataset en el proyecto, es necesario descargar tanto la carpeta `annotations/` con las etiquetas de las imágenes como la carpeta `images/`. Al descargar se genera un ZIP para las etiquetas y entre 3-4 ZIP para las imágenes, hay que extraerlo todo y añadirlo al repositorio. Será necesario un directorio `images/` para todas las imágenes y otro `labels/` para las etiquetas.

## Detección de matrículas

### Entorno para YOLO con Python 3.9.25

Dentro del entorno de Anaconda se instalaron las siguientes para el entrenamiento del modelo YOLO:

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

Además, la GPU utilizada para el entrenamiento fue la siguiente:

- NVIDIA GeForce RTX 4050 Laptop (driver 617.42), CUDA 12.6.

### Reparto de imágenes del dataset

Con el entorno de Anaconda iniciado y el dataset en el repositorio, se procede con la ejecución del código para el entrenamiento y prueba del modelo YOLO en la detección de matrículas, ubicado en el cuaderno `YOLO_P4.ipynb`.

La primera celda de código está destinada al reparto de las imágenes del dataset, por lo que solo será necesario ejecutarlo la primera vez que se importa. Este genera un nuevo directorio llamado `dataset/`, el cual reparte todas las imagenes en otros tres subdirectorios `train/` (80%), `test/` (10%) y `val/` (10%).

El código va emparejando cada imagen con su etiqueta buscando los ficheros que contengan el mismo nombre, para posteriormente mezclar estas parejas aleatoriamente a través de una semilla en cada subdirectorio.

### Entrenamiento del modelo YOLO (`YOLO_P4.ipynb`)

Una vez generado el directorio `dataset/`, se procede con la importación de los paquetes en el código para la ejecución del entrenamiento y manejo de YOLO. Aunque es necesario primero convertir las etiquetas de `.json` a `.txt` para que el modelo pueda usarlas. 

#### Configuración del modelo YOLO

Posteriormente, se configura el modelo para ejecutar el entrenamiento correctamente con los siguientes parámetros: 

- `path`, para localizar el directorio `dataset/`; `train` para las imágenes de entrenamiento en `dataset/train/`, con las que el modelo irá aprendiendo en cada época.

- `val` para las imágenes de evaluación en `dataset/val/`, estas imágenes permiten realizar una pequeña prueba al final de cada época para evaluar el rendimiento del modelo, permitiendo realizar ajustes a partir de sus resultados para la siguiente época.

- `test` para las imágenes de prueba final en directorio `dataset/test/`, con las que se probará el modelo final alcanzado en la última época, su objetivo es valorar el rendimiento que tendrá el modelo cuando se empiece a probar con imágenes nuevas.

- `names` cuyo valor es `plate` para el tipo `0`. Esto se traduce en que cuando el modelo detecte un objeto lo llamará `plate`. El `0` indica que se trata del primer objeto que busca, al ser un modelo destinado únicamente a matrículas no es necesario añadir ningun número más.

#### Entrenamiento del modelo YOLO

Con la configuración realizada, se procede con la función `train()`. Para este modelo y dataset se ejecutaron dos entrenamientos: uno simple y otro con `data augmentation`. Ambos entrenamientos parten de 100 épocas con 20 de paciencia, que se traduce en que si no se ve una mejora del modelo en máximo 20 épocas, se detiene el entrenamiento. 

Por otro lado, el parámetro `imgz` se utiliza para reescalar las imagenes a 960px. Este valor se debe a que al intentar utilizar imágenes más pequeñas para agilizar el entrenamiento, se podían volver irreconocibles las matrículas más pequeñas, generando falsos negativos. Y de otra forma, si se intentaba aumentar el tamaño a 1280px se desbordaba la memoria de mi GPU. 

El parámetro `batch` indica el número de imágenes que se procesan a la vez, aumentar o reducir este valor es importante para evitar los desbordamientos de memoria y agilizar lo máximo posible el entrenamiento. En mi caso, utilicé un valor de `4` ya que aumentarlo me generaba error en la memoria y reducirlo volvía demasiado lento el entrenamiento.

Finalmente, los resultados se guardan en un nuevo directorio llamado `results`, para el entrenamiento base utilicé el nombre `plate-detection` en el parámetro `name`; mientras que para el segundo entrenamiento con `data augmentation` utilicé `plate-detection-augmentation`.

En el caso de `plate-detection-augmentation`, que es el que se muestra en el cuaderno, los parámetros utilizados fueron los de la siguiente tabla:

| Parámetro | Defecto | Tipo | Categoría | Qué hace | ¿Aplica a detección? |
|---|---|---|---|---|---|
| `hsv_h` | 0.015 | float | Color | Cambia el tono (color base) de la imagen, hasta ±1.5 % de la rueda de color. | Sí |
| `hsv_s` | 0.7 | float | Color | Cambia la saturación, hasta ±70 %. | Sí |
| `hsv_v` | 0.4 | float | Color | Cambia el brillo, hasta ±40 %. | Sí |
| `bgr` | 0.0 | float (prob.) | Color | Probabilidad de intercambiar los canales RGB por BGR. | Sí |
| `degrees` | 0.0 | float | Geometría | Rota la imagen hasta ±N grados. | Sí |
| `translate` | 0.1 | float | Geometría | Desplaza la imagen hasta ±10 % de su tamaño. | Sí |
| `scale` | 0.5 | float o (min, max) | Geometría | Agranda o encoge la imagen hasta ±50 %. | Sí |
| `shear` | 0.0 | float | Geometría | Inclina la imagen en diagonal, hasta ±N grados. | Sí |
| `perspective` | 0.0 | float | Geometría | Simula un cambio de punto de vista (valores típicos de 0 a 0.001). | Sí |
| `fliplr` | 0.5 | float (prob.) | Geometría | Probabilidad de voltear la imagen horizontalmente (espejo). | Sí |
| `flipud` | 0.0 | float (prob.) | Geometría | Probabilidad de voltear la imagen verticalmente. | Sí |
| `mosaic` | 1.0 | float (prob.) | Mezcla | Probabilidad de combinar 4 imágenes en una. | Sí |
| `mixup` | 0.0 | float (prob.) | Mezcla | Probabilidad de superponer dos imágenes con transparencia. | Sí |
| `cutmix` | 0.0 | float (prob.) | Mezcla | Probabilidad de recortar un trozo de una imagen y pegarlo en otra. | Sí |
| `close_mosaic` | 10 | int (épocas) | Mezcla | Desactiva el mosaico en las últimas N épocas del entrenamiento. | Sí |
| `multi_scale` | 0.0 | float | Escala | Varía el tamaño de entrada durante el entrenamiento, como fracción de `imgsz`. | Sí |
| `copy_paste` | 0.0 | float | Mezcla | Copia objetos de unas imágenes a otras (segmentación y OBB). | No |
| `copy_paste_mode` | `flip` | str | Mezcla | Estrategia de copy-paste: `flip` o `mixup`. | No |
| `auto_augment` | `randaugment` | str | Política | Política de aumento automático: `randaugment`, `autoaugment` o `augmix`. | No (clasificación) |
| `erasing` | 0.4 | float (prob.) | Política | Probabilidad de borrar un trozo aleatorio de la imagen. | No (clasificación) |

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
