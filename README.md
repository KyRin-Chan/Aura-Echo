# Voice Changer (RTX 5080 Optimized Edition)

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [How to Use (Pre-built Release)](#how-to-use-pre-built-release)
  - [Prerequisites](#prerequisites)
  - [Unpacking and Running](#unpacking-and-running)
- [Troubleshooting](#troubleshooting)
  - [Download verification errors](#exceptionspretraindownloadexception-failed-to-download-weight)
  - [Audio devices are not displayed](#audio-devices-are-not-displayed)
  - [No sound after start](#no-sound-after-start)
  - [Hearing non-converted voice](#hearing-non-converted-voice)
  - [Hearing audio crackles](#hearing-audio-crackles)
  - [Audio is stuttery](#audio-is-stuttery)
- [Development / Working with Source](#working-with-the-source)
  - [Setting up the environment](#setting-up-the-environment)
  - [Running the server](#running-the-server)
  - [Building a package](#building-a-package)

---

## Overview

This is a customized fork of [deiteris voice changer](https://github.com/deiteris/voice-changer) designed for real-time voice conversion using Retrieval-based Voice Conversion (RVC).

> [!IMPORTANT]
> This edition is specifically optimized for **Windows 10/11** and **Nvidia GPUs (CUDA)**, with dedicated performance tunings for **RTX 5080 / Blackwell architecture (TensorRT / ONNX)**. Other backends (DirectML, CPU-only, ROCm) and operating systems (macOS, Linux) are not supported in this branch.

---

## System Requirements

- **Operating System**: Windows 10 or Windows 11 (64-bit).
- **RAM**: At least 6GB.
- **Disk Space**: At least 6GB of free space (SSD recommended).
- **GPU**: Nvidia GeForce RTX 20 Series or later (Optimized for RTX 5080).
- **GPU VRAM**: At least 2GB (FP32) or 1GB (FP16).
- **Nvidia Driver**: Version `528.33` or later.

---

## How to Use (Pre-built Release)

### Prerequisites

1. Download and install [7-Zip](https://www.7-zip.org/) or [WinRAR](https://www.win-rar.com/).
2. Download and install [VAC Lite by Muzychenko](https://software.muzychenko.net/freeware/vac470lite.zip) or another virtual audio cable driver.
3. Download the latest release package (`voice-changer-windows-amd64-cuda-5080.7z`) from the Releases section.

### Unpacking and Running

1. Put the downloaded `.7z` file in your preferred directory.
2. Right-click and select **7-Zip** > **Extract to "voice-changer-windows-amd64-cuda-5080\"**.
3. Open the extracted folder, navigate to `MMVCServerSIO`.
4. Run `MMVCServerSIO.exe`.
   - *Note: On first startup, it will automatically download necessary pre-trained weights. Do not close the command prompt until the download finishes.*
5. Once the weights are downloaded, your default web browser will open the graphical interface.

---

## Troubleshooting

> [!TIP]
> When any issue occurs, check the command prompt window for specific error output.

### Exceptions.PretrainDownloadException: 'Failed to download weight.'
If files were corrupted or interrupted during download, check the terminal for lines like:
```text
[WeightDownloader] 'pretrain/content_vec_500.onnx failed to pass hash verification check.'
```
Go to the voice changer folder, locate the failed file under the `pretrain` directory, delete it, and restart the voice changer to re-download.

### Audio devices are not displayed
1. Make sure you have granted microphone access permissions in Windows Settings.
2. Use Google Chrome or a Chromium-based browser. Firefox ESR is not recommended.

### No sound after start
1. Verify that the correct input and output audio devices are selected in the web UI.
2. Verify that your hardware microphone is not muted and the volume is up.

### Hearing non-converted voice
Make sure **passthru** is off in the web UI (the "Passthrough On" button is yellow when active; click it to turn it off).

### Hearing audio crackles
1. Make sure you are using a Virtual Audio Cable (e.g. **Line 1**).
2. Ensure the sample rate of your microphone matches the sample rate of the Virtual Audio Cable in the Windows **Sound Control Panel**.
3. If crackling persists, open **Task Manager** > **Details**, find `audiodg.exe`:
   - Right-click `audiodg.exe` > **Set priority** > **High**.
   - Right-click `audiodg.exe` > **Set affinity** > Uncheck all, and select only **CPU 2**.

### Audio is stuttery
1. If you changed settings (Chunk, Extra, Crossfade) while voice conversion was active, click **Stop** and then **Start**.
2. Make sure the reported `perf` conversion time in the UI is smaller than the selected `Chunk` size. If `perf` exceeds the chunk size, increase `Chunk` size or reduce `Extra` and `Crossfade` settings.

---

## Working with the Source

### Setting up the environment

1. Install [Python 3.12](https://www.python.org/downloads/) (make sure to add Python to PATH).
2. Install [Git](https://git-scm.com/).
3. Clone this repository:
   ```cmd
   git clone <repo-url>
   cd rvc-custom/server
   ```
4. Run the installation script to set up a virtual environment and install RTX 5080 CUDA requirements:
   ```cmd
   .\vc_install.bat
   ```

### Running the server

To launch the development server:
```cmd
.\vc_startup.bat
```
Then open the browser and navigate to the address shown in the command line (usually `http://localhost:18950`).

### Building a package

To package the project into a standalone executable:
```cmd
pip install --upgrade pip wheel setuptools pyinstaller
pyinstaller --clean -y --dist ./dist --workpath /tmp MMVCServerSIO.spec
```
The compiled package will be outputted to the `dist` folder.
