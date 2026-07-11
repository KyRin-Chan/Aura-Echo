@echo off
setlocal enabledelayedexpansion

nvidia-smi --query-supported-clocks=graphics,memory --format=csv,noheader,nounits > temp_clocks.txt 2>nul

if not exist temp_clocks.txt (
    echo [ERROR] Failed to query supported clocks using nvidia-smi.
    echo Please make sure NVIDIA drivers are installed and nvidia-smi is in your PATH.
    pause
    exit /b 1
)

REM Read the first line from the temporary file for maximum clocks
set /p firstLine=<temp_clocks.txt
if "%firstLine%"=="" (
    echo [ERROR] No supported clocks found.
    del temp_clocks.txt
    pause
    exit /b 1
)

for /f "tokens=1,2 delims=, " %%a in ("%firstLine%") do (
    set max_gpu=%%a
    set max_mem=%%b
)

echo ==========================================================
echo               NVIDIA GPU Clock Lock Utility
echo ==========================================================
echo  [1] Lock to MAXIMUM clocks (GPU: %max_gpu% MHz, Mem: %max_mem% MHz)
echo  [2] Manually enter custom frequencies
echo  [3] Choose from supported clocks list
echo ==========================================================
set /p choice="Select option (1-3) [Default 1]: "

if "%choice%"=="" set choice=1
if "%choice%"=="1" (
    set gpu_clock=%max_gpu%
    set memory_clock=%max_mem%
    goto lock_clocks
)
if "%choice%"=="2" (
    goto enter_clocks
)
if "%choice%"=="3" (
    goto list_clocks
)
echo Invalid option selected. Defaulting to maximum clocks.
set gpu_clock=%max_gpu%
set memory_clock=%max_mem%
goto lock_clocks

:enter_clocks
echo.
echo Please specify frequencies in MHz (e.g., GPU: 1500, Memory: 5001)
set /p gpu_clock="Target GPU Clock (MHz): "
set /p memory_clock="Target Memory Clock (MHz): "
if "%gpu_clock%"=="" (
    echo GPU Clock cannot be empty.
    del temp_clocks.txt
    pause
    exit /b 1
)
if "%memory_clock%"=="" (
    echo Memory Clock cannot be empty.
    del temp_clocks.txt
    pause
    exit /b 1
)
goto lock_clocks

:list_clocks
echo.
echo Supported clock combinations (showing top 30):
echo ----------------------------------------------------------
set count=0
for /f "tokens=1,2 delims=, " %%a in (temp_clocks.txt) do (
    set /a count+=1
    set "clk_gpu[!count!]=%%a"
    set "clk_mem[!count!]=%%b"
    echo [!count!] GPU: %%a MHz, Memory: %%b MHz
    if !count! geq 30 (
        echo ... (List truncated, showing top 30)
        goto choose_from_list
    )
)

:choose_from_list
echo ----------------------------------------------------------
set /p clk_choice="Select clock combination number (1-!count!): "
if "%clk_choice%"=="" (
    echo No selection made.
    del temp_clocks.txt
    pause
    exit /b 1
)
for %%i in (!clk_choice!) do (
    set gpu_clock=!clk_gpu[%%i]!
    set memory_clock=!clk_mem[%%i]!
)

if "!gpu_clock!"=="" (
    echo Invalid selection index.
    del temp_clocks.txt
    pause
    exit /b 1
)
goto lock_clocks

:lock_clocks
if exist temp_clocks.txt del temp_clocks.txt

echo.
echo Locking GPU clock to %gpu_clock% MHz...
nvidia-smi --lock-gpu-clocks=%gpu_clock%
if %errorlevel% neq 0 (
    echo [WARNING] Failed to lock GPU clock. nvidia-smi may require Administrator privileges.
)

echo Locking Memory clock to %memory_clock% MHz...
nvidia-smi --lock-memory-clocks=%memory_clock%
if %errorlevel% neq 0 (
    echo [WARNING] Failed to lock Memory clock. nvidia-smi may require Administrator privileges.
)

echo.
echo Operation completed.
pause