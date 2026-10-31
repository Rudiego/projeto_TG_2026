# Detector de drones e pássaros com YOLO

Projeto de detecção de drones e pássaros com YOLO. O treinamento e a avaliação são executados no computador; para o Raspberry Pi são enviados apenas o modelo exportado e o programa de inferência.

O dataset, os resultados de treinamento, os pesos e os modelos exportados são mantidos localmente e não são enviados ao GitHub.

## Estrutura do projeto

```text
projeto_TG_2026/
├── config/
│   └── project.yaml              # Configurações, classes e hiperparâmetros
├── datasets/
│   └── drone_bird/               # Dataset local (ignorado pelo Git)
├── models/
│   └── exported/                 # Modelos exportados (ignorado pelo Git)
├── raspberry/
│   └── inference_camera.py       # Inferência com Picamera2
├── runs/                         # Treinos, métricas e previsões (ignorado pelo Git)
├── scripts/
│   ├── check_dataset.py          # Validação do dataset
│   ├── train.py                  # Treinamento
│   ├── evaluate.py               # Avaliação
│   ├── predict.py                # Predição local
│   └── export_raspberry.py       # Exportação do modelo
├── src/drone_bird_yolo/          # Código reutilizável do projeto
├── tests/                        # Testes automatizados
├── requirements.txt              # Dependências do computador
└── requirements-raspberry.txt    # Dependências do Raspberry Pi
```

## Organização do dataset

Crie as pastas abaixo dentro do projeto:

```text
datasets/drone_bird/
├── images/
│   ├── train/
│   │   ├── imagem_001.jpg
│   │   └── imagem_002.jpg
│   └── test/
│       └── imagem_101.jpg
└── labels/
    ├── train/
    │   ├── imagem_001.txt
    │   └── imagem_002.txt
    └── test/
        └── imagem_101.txt
```

Cada imagem deve ficar na mesma divisão (`train` ou `test`) que seu label e ambos devem ter o mesmo nome-base. Por exemplo:

```text
images/train/drone_001.jpg
labels/train/drone_001.txt
```

Os labels usam o formato YOLO. Cada linha representa um objeto:

```text
classe x_centro y_centro largura altura
```

Exemplo de uma caixa da classe `drone`:

```text
0 0.512 0.438 0.210 0.165
```

Os valores das coordenadas e dimensões devem estar normalizados entre `0` e `1`. As classes configuradas em `config/project.yaml` são:

```text
0 = drone
1 = bird
```

Uma imagem sem nenhum objeto pode ficar sem arquivo `.txt`; ela será tratada como imagem de fundo. Use a validação estrita caso queira exigir um `.txt` para todas as imagens.

> O conjunto `test` é usado durante o treinamento para acompanhar as métricas. Para a avaliação final do TG, mantenha também um conjunto externo que nunca seja usado durante o treinamento.

## Preparação do ambiente no Windows

Na pasta do projeto, execute:

```powershell
py -3.13 -m venv .venv --system-site-packages
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Comandos principais

### Validar o dataset

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py
```

Para exigir um arquivo de label para cada imagem:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py --strict
```

O verificador detecta, entre outros problemas, labels órfãos, classes inválidas, coordenadas fora do intervalo permitido e caixas sem área.

### Executar os testes

```powershell
.\.venv\Scripts\python.exe -m pytest
```

### Treinar

```powershell
.\.venv\Scripts\python.exe scripts\train.py
```

As configurações padrão ficam em `config/project.yaml`. Também é possível sobrescrevê-las pela linha de comando:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --epochs 100 --imgsz 640 --batch 4 --device 0
```

Se faltar memória de GPU, reduza o lote:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --batch 2
```

Para retomar um treinamento interrompido:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --resume runs\detect\train_drone_bird\weights\last.pt
```

### Avaliar o modelo

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py
```

Para escolher um peso específico:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --model runs\detect\train_drone_bird\weights\best.pt
```

### Fazer uma predição

Imagem:

```powershell
.\.venv\Scripts\python.exe scripts\predict.py caminho\imagem.jpg
```

Vídeo:

```powershell
.\.venv\Scripts\python.exe scripts\predict.py caminho\video.mp4
```

Webcam:

```powershell
.\.venv\Scripts\python.exe scripts\predict.py 0 --show
```

Para escolher o modelo e o limite de confiança:

```powershell
.\.venv\Scripts\python.exe scripts\predict.py caminho\imagem.jpg --model runs\detect\train_drone_bird\weights\best.pt --conf 0.45
```

### Exportar para o Raspberry Pi

O formato padrão é NCNN:

```powershell
.\.venv\Scripts\python.exe scripts\export_raspberry.py
```

Para exportar um peso específico em FP16:

```powershell
.\.venv\Scripts\python.exe scripts\export_raspberry.py --model runs\detect\train_drone_bird\weights\best.pt --fp16
```

Também estão disponíveis os formatos `onnx` e `openvino`:

```powershell
.\.venv\Scripts\python.exe scripts\export_raspberry.py --format onnx
```

## Execução no Raspberry Pi

Copie para o Raspberry apenas:

- a pasta do modelo NCNN gerada em `models/exported/`;
- `raspberry/inference_camera.py`;
- `requirements-raspberry.txt`.

No Raspberry Pi OS:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-venv
python3 -m venv .venv --system-site-packages
.venv/bin/python -m pip install -r requirements-raspberry.txt
.venv/bin/python raspberry/inference_camera.py --model models/best_ncnn_model
```

No Raspberry Pi OS Lite, execute sem interface gráfica:

```bash
.venv/bin/python raspberry/inference_camera.py --model models/best_ncnn_model --headless
```

Para reduzir o uso de CPU, use `--process-every 2` ou diminua `--imgsz`.

## Arquivos que não vão para o GitHub

O `.gitignore` impede o envio de:

- imagens e labels do dataset;
- backups locais do dataset;
- resultados da pasta `runs/`;
- pesos e modelos exportados (`.pt`, `.onnx`, `.torchscript`);
- ambiente virtual `.venv/`;
- caches, auditorias e arquivos temporários.

Assim, outra pessoa recebe o código e as instruções, mas precisa obter o dataset e os pesos separadamente.
