<div align="center">

# AuraEcho

**Real-time voice conversion powered by RVC, tuned for CUDA on Windows.**

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D6?logo=windows)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-Non--Commercial-orange)

[Overview](#overview) · [Quick Start](#quick-start) · [Troubleshooting](#troubleshooting) · [Development](#development)

</div>

---

## Overview

AuraEcho provides low-latency, real-time voice conversion based on Retrieval-based Voice Conversion (RVC).

This edition focuses on a single, well-tuned configuration for **NVIDIA RTX GPUs**, with acceleration via **CUDA / TensorRT / ONNX**.

> [!IMPORTANT]
> Only **Windows 10/11** with **NVIDIA GPUs (CUDA)** is supported. DirectML, CPU-only, ROCm, macOS, and Linux are not tested on this branch.

## System Requirements

| Component      | Requirement                                              |
| -------------- | -------------------------------------------------------- |
| OS             | Windows 10 / 11 (64-bit)                                 |
| GPU            | NVIDIA GeForce RTX 20 series or newer                    |
| VRAM           | ≥ 2 GB (FP32) or ≥ 1 GB (FP16)                           |
| NVIDIA driver  | 528.33 or later                                          |
| RAM            | ≥ 6 GB                                                   |
| Disk           | ≥ 6 GB free (SSD recommended)                            |

## Quick Start

### 1. Prerequisites

- An archiver such as [7-Zip](https://www.7-zip.org/) or [WinRAR](https://www.win-rar.com/).
- A virtual audio cable, e.g. [VAC Lite by Muzychenko](https://software.muzychenko.net/freeware/vac470lite.zip).
- The latest release package, `AuraEcho-windows-amd64-cuda.7z`, from the **Releases** page.

### 2. Run

1. Extract the `.7z` archive to a directory of your choice.
2. Open the extracted folder, then the `AuraEcho` directory.
3. Launch `AuraEcho.exe`.
4. Wait for the initial download of pre-trained weights to finish. Keep the console window open until it completes.
5. Your default browser will open the web interface automatically.

## Troubleshooting

> [!TIP]
> When something goes wrong, the console window usually shows the relevant error message first.

<details>
<summary><b>Failed to download weight (<code>PretrainDownloadException</code>)</b></summary>

A corrupted or interrupted download produces a message such as:

```text
[WeightDownloader] 'pretrain/content_vec_500.onnx failed to pass hash verification check.'
```

Delete the failing file from the `pretrain` directory and restart the application to download it again.

</details>

<details>
<summary><b>Audio devices are not listed</b></summary>

1. Grant microphone access in Windows **Settings**.
2. Use Google Chrome or another Chromium-based browser (Firefox ESR is not recommended).

</details>

<details>
<summary><b>No sound after starting</b></summary>

1. Confirm the correct input and output devices are selected in the web UI.
2. Make sure your microphone is not muted and its volume is raised.

</details>

<details>
<summary><b>Hearing the unconverted voice</b></summary>

Turn **Passthrough** off in the web UI. The button is yellow while active; click it to disable.

</details>

<details>
<summary><b>Audio crackling</b></summary>

1. Route audio through a virtual audio cable (e.g. **Line 1**).
2. In the Windows **Sound** control panel, make sure the microphone and the virtual cable use the same sample rate.
3. If crackling persists, open **Task Manager → Details** and locate `audiodg.exe`:
   - Right-click → **Set priority** → **High**.
   - Right-click → **Set affinity** → clear all, then select only **CPU 2**.

</details>

<details>
<summary><b>Stuttering audio</b></summary>

1. After changing **Chunk**, **Extra**, or **Crossfade** during conversion, click **Stop** and then **Start**.
2. Ensure the `perf` time shown in the UI is lower than the selected **Chunk** size. Otherwise, increase **Chunk** or reduce **Extra** and **Crossfade**.

</details>

## Development

### Set up the environment

1. Install [Python 3.12](https://www.python.org/downloads/) (add it to `PATH`) and [Git](https://git-scm.com/).
2. Clone the repository and run the installer, which creates a virtual environment and installs the CUDA dependencies:

   ```cmd
   git clone https://github.com/KyRin-Chan/Aura-Echo.git
   cd Aura-Echo/server
   .\vc_install.bat
   ```

### Run the development server

```cmd
.\vc_startup.bat
```

Then open the address printed in the console (typically `http://localhost:18950`).

### Build a standalone package

```cmd
pip install --upgrade pip wheel setuptools pyinstaller
pyinstaller --clean -y --dist ./dist --workpath /tmp AuraEcho.spec
```

The output is written to the `dist` directory.

## License

AuraEcho is released under a [Non-Commercial License](LICENSE) — free for personal and non-commercial use; commercial use and paid redistribution are prohibited without prior written permission. Upstream third-party components remain under their original MIT Licenses as detailed in [LICENSE](LICENSE).

## Acknowledgements

Built upon [deiteris/voice-changer](https://github.com/deiteris/voice-changer) and the original [voice-changer](https://github.com/w-okada/voice-changer) by Wataru Okada.
