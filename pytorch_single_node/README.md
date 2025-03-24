# PyTorch на SLURM с использованием NVIDIA Enroot контейнера

В этом репозитории представлены примеры обучения свёрточной нейронной сети на наборе данных MNIST с помощью PyTorch и Slurm, с использованием различных подходов к параллельному обучению:

1. Обучение на одном узле с одним GPU
2. Обучение на одном узле с несколькими GPU (DataParallel)
3. Распределенное обучение на нескольких узлах (DistributedDataParallel)

## Предварительные требования
- Доступ к кластеру HPC с менеджером задач **Slurm**
- Поддержка контейнеров **Enroot**
- GPU с поддержкой CUDA
- Общая файловая система (`/scratch/$USER/`) для хранения данных и кода

## Структура репозитория
```
pytorch_single_node/
├── train_mnist.py                # Базовый скрипт для обучения CNN на MNIST (одиночный GPU)
├── train_mnist_dataparallel.py   # Скрипт с поддержкой DataParallel (несколько GPU на одном узле)
├── train_mnist_distributed.py    # Скрипт с поддержкой DistributedDataParallel (несколько узлов)
├── single_node.slurm             # Скрипт Slurm для одного GPU
├── dataparallel.slurm            # Скрипт Slurm для DataParallel
├── multi_node.slurm              # Скрипт Slurm для распределенного обучения
├── resume_training.slurm         # Скрипт для возобновления обучения (одиночный GPU)
├── resume_dataparallel.slurm     # Скрипт для возобновления обучения (DataParallel)
├── resume_distributed.slurm      # Скрипт для возобновления распределенного обучения
└── README.md                     # Этот файл
```

---

## **1. Настройка путей для данных**

В скриптах SLURM вы можете изменить следующие переменные для настройки путей:

```bash
# Путь к данным на хосте
DATA_DIR="/scratch/${USER}/datasets/mnist"

# Путь к данным внутри контейнера
CONTAINER_DATA_DIR="/workspace/datasets/mnist"
```

Директории будут автоматически созданы при запуске скрипта, если они не существуют.

---

## **2. Варианты параллельного обучения**

### **2.1 Одиночный GPU (Базовый вариант)**

Самый простой вариант – запуск на одном узле с одним GPU:

```bash
cd /scratch/$USER/msu_slurm_test/pytorch_single_node
sbatch single_node.slurm
```

### **2.2 DataParallel (Несколько GPU на одном узле)**

PyTorch DataParallel позволяет использовать несколько GPU на одном узле, автоматически разделяя батчи между доступными устройствами:

```bash
cd /scratch/$USER/msu_slurm_test/pytorch_single_node
sbatch dataparallel.slurm
```

Особенности DataParallel:
- Простота использования - достаточно обернуть модель в `nn.DataParallel`
- Работает только на одном узле
- Есть некоторые накладные расходы на коммуникацию между GPU
- Все параметры модели дублируются на каждом GPU

### **2.3 DistributedDataParallel (Несколько узлов, несколько GPU)**

Для масштабирования на несколько узлов используется `torch.distributed` и `DistributedDataParallel`:

```bash
cd /scratch/$USER/msu_slurm_test/pytorch_single_node
sbatch multi_node.slurm
```

Особенности DistributedDataParallel:
- Более эффективное использование нескольких GPU
- Масштабируется на несколько узлов
- Каждый процесс работает независимо, синхронизируя только градиенты
- Требует использования `DistributedSampler` для правильного разделения данных

---

## **3. Возобновление обучения из чекпоинта**

### **3.1 Для одиночного GPU:**

```bash
cd /scratch/$USER/msu_slurm_test/pytorch_single_node
sbatch resume_training.slurm
```

### **3.2 Для DataParallel (несколько GPU на одном узле):**

```bash
cd /scratch/$USER/msu_slurm_test/pytorch_single_node
sbatch resume_dataparallel.slurm
```

В скрипте `resume_dataparallel.slurm` можно изменить переменную `CHECKPOINT_FILE` для указания нужного чекпоинта:

```bash
CHECKPOINT_FILE="checkpoint_epoch_10.pt"  # Измените на нужный чекпоинт
```

### **3.3 Для распределенного обучения:**

```bash
cd /scratch/$USER/msu_slurm_test/pytorch_single_node
sbatch resume_distributed.slurm
```

В скрипте `resume_distributed.slurm` можно изменить переменную `CHECKPOINT_FILE`, чтобы указать конкретный чекпоинт для продолжения обучения.

---

## **Особенности использования Enroot**

В скриптах SLURM используется контейнер NVIDIA PyTorch:
```
nvcr.io+nvidia+pytorch+24.04-py3.sqsh
```

Основные флаги:
- `--container-image`: указывает на образ контейнера
- `--container-mounts`: монтирует директории хоста в контейнер
- `--container-workdir`: устанавливает рабочую директорию внутри контейнера

---

## **Полезные команды**
- `squeue -u $USER` - список всех запущенных Ваших работ
- `sinfo` - информация о доступных узлах и разделах
- `scontrol show job <jobid>` - подробная информация о задании
- `scancel <jobid>` - отмена задания
