# Dev Environment: Visão computacional

## Passthrough da webcam para o WSL2

> Executar em PowerShell (Admin)

Listagem dos dispositivos USB
```console
usbipd list
```

### Para preparar e conectar o dispositivo:

1. Fazer bind do dispositivo
```console
usbipd bind --busid <busid>
```

2. Conectar o dispositivo
```console
usbipd attach --wsl --busid <busid>
```

> Desconectar o dispositivo
> ```console
> usbipd detach --busid <busid>
> ```

## Verifique se a webcam está disponível

> Executar no WSL

```console
ls /dev/video*
```

Teste com `ffmpeg`

```console
ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1280x720 -i /dev/video0 -vframes 1 test.jpg
```

## Python

```console
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```