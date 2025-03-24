# MPI  "Hallo, world" с помощью Slurm

В этом репозитории представлен простой пример MPI "Hallo, world" для запуска на кластере HPC с помощью Slurm. Запуск включает в себя:
- компиляцию
- запуск программы на одном узле (`single_node.slurm`)
- запуск программы на нескольких узлах (`multi_node.slurm`)
- примеры различных стратегий привязки процессов (`binding_examples.slurm`)

## Предварительные требования
- Доступ к кластеру HPC с помощью **Slurm**
- Общая файловая система (`/scratch/$USER/`) для хранения файлов

## Структура репозитория
```
mpi_hello_world/
├── hello_mpi.c           # Простая программа MPI
├── Makefile              # Инструкции по компиляции
├── single_node.slurm     # Скрипт Slurm для одноузлового выполнения
├── multi_node.slurm      # Скрипт Slurm для многоузлового выполнения
└── README.md             # Этот файл
```

---

## **1. Компиляция**
Чтобы скомпилировать программу MPI просто используем mpicc. Переходи в папку `mpi_hello_world` и запускаем `make`

---

## **2. Запуск на одном узле**
После компиляции вы можете запустить программу на одном узле с помощью:
```bash
sbatch single_node.slurm
```
Этот скрипт:
- запрашивает 1 узел с 4 процессами MPI
- привязывает каждый процесс к отдельному ядру
- запускает скомпилированную программу `hello_mpi`

### **Проверка результатов выполнения**
```bash
cat single_node_<job_id>.out
```

Пример вывода (все процессы на одном узле):
```
Hello world from processor cn2.sc.test, rank 0 out of 4 processors
Hello world from processor cn2.sc.test, rank 2 out of 4 processors
Hello world from processor cn2.sc.test, rank 3 out of 4 processors
Hello world from processor cn2.sc.test, rank 1 out of 4 processors
```

---

## **3. Запуск на нескольких узлах**
Для запуска на нескольких узлах:
```bash
sbatch multi_node.slurm
```
Этот скрипт:
- запрашивает 2 узла с 20 процессами (по 10 на каждый узел)
- использует привязку к ядрам для оптимальной производительности
- использует `srun --mpi=pmix` для запуска процессов MPI на разных узлах

Пример вывода (0-9 попали на cn2.sc.test, а 10-19 на cn3.sc.test):
```
Hello world from processor cn3.sc.test, rank 10 out of 20 processors
Hello world from processor cn2.sc.test, rank 8 out of 20 processors
Hello world from processor cn3.sc.test, rank 12 out of 20 processors
Hello world from processor cn3.sc.test, rank 16 out of 20 processors
Hello world from processor cn3.sc.test, rank 18 out of 20 processors
Hello world from processor cn3.sc.test, rank 19 out of 20 processors
Hello world from processor cn3.sc.test, rank 11 out of 20 processors
Hello world from processor cn3.sc.test, rank 13 out of 20 processors
Hello world from processor cn3.sc.test, rank 14 out of 20 processors
Hello world from processor cn3.sc.test, rank 15 out of 20 processors
Hello world from processor cn3.sc.test, rank 17 out of 20 processors
Hello world from processor cn2.sc.test, rank 9 out of 20 processors
Hello world from processor cn2.sc.test, rank 3 out of 20 processors
Hello world from processor cn2.sc.test, rank 7 out of 20 processors
Hello world from processor cn2.sc.test, rank 0 out of 20 processors
Hello world from processor cn2.sc.test, rank 2 out of 20 processors
Hello world from processor cn2.sc.test, rank 4 out of 20 processors
Hello world from processor cn2.sc.test, rank 1 out of 20 processors
Hello world from processor cn2.sc.test, rank 6 out of 20 processors
Hello world from processor cn2.sc.test, rank 5 out of 20 processors
```

### **Проверка вывода результатов выполнения**
```bash
cat multi_node_<job_id>.out
```

## **Полезные команды**
- `squeue -u $USER` - список всех запущенных Ваших работ
- `sinfo` - информация о доступных узлах и разделах
- `scontrol show job <jobid>` - подробная информация о задании
- `scancel <jobid>` - отмена задания


## Подробнее о привязке процессов к ядрам/узлам
- `man sbatch`
- `man srun`
