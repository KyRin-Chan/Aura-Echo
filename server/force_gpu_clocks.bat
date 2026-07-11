@echo off
setlocal enabledelayedexpansion

REM Query supported memory clocks to temp_mem.txt
nvidia-smi --query-supported-clocks=memory --format=csv,noheader,nounits > temp_mem.txt 2>nul

if not exist temp_mem.txt (
    echo [ERROR] Failed to query supported clocks using nvidia-smi.
    echo Please make sure NVIDIA drivers are installed and nvidia-smi is in your PATH.
    pause
    exit /b 1
)

REM Query all supported clock combinations
nvidia-smi --query-supported-clocks=graphics,memory --format=csv,noheader,nounits > temp_all.txt 2>nul

REM Read maximum clocks from the first line of temp_all.txt
set /p firstLine=<temp_all.txt
if "%firstLine%"=="" (
    echo [ERROR] No supported clocks found.
    if exist temp_mem.txt del temp_mem.txt
    if exist temp_all.txt del temp_all.txt
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
    goto list_mem_clocks
)
echo Invalid option selected. Defaulting to maximum clocks.
set gpu_clock=%max_gpu%
set memory_clock=%max_mem%
goto lock_clocks

:enter_clocks
echo.
echo Please specify frequencies in MHz (e.g., GPU: 1500, Memory: 9001)
set /p gpu_clock="Target GPU Clock (MHz): "
set /p memory_clock="Target Memory Clock (MHz): "
if "%gpu_clock%"=="" (
    echo GPU Clock cannot be empty.
    goto cleanup_and_exit
)
if "%memory_clock%"=="" (
    echo Memory Clock cannot be empty.
    goto cleanup_and_exit
)
goto lock_clocks

:list_mem_clocks
echo.
echo Supported Memory Clocks:
echo ----------------------------------------------------------
set mem_count=0
for /f "tokens=1" %%m in (temp_mem.txt) do (
    set /a mem_count+=1
    set "mem_val[!mem_count!]=%%m"
    echo [!mem_count!] %%m MHz
)
echo ----------------------------------------------------------
set /p mem_choice="Select Memory Clock (1-!mem_count!): "
if "%mem_choice%"=="" (
    echo No selection made.
    goto cleanup_and_exit
)

set chosen_mem=
for %%i in (!mem_choice!) do (
    if defined mem_val[%%i] (
        set chosen_mem=!mem_val[%%i]!
    )
)

if "!chosen_mem!"=="" (
    echo Invalid Memory Clock selection.
    goto cleanup_and_exit
)

REM Now filter graphics clocks for the chosen memory clock
set gpu_count=0
for /f "tokens=1,2 delims=, " %%g in (temp_all.txt) do (
    if "%%h"=="%chosen_mem%" (
        set /a gpu_count+=1
        set "gpu_val[!gpu_count!]=%%g"
    )
)

if !gpu_count! equ 0 (
    echo No supported GPU clocks found for Memory clock %chosen_mem% MHz.
    goto cleanup_and_exit
)

echo.
echo Supported GPU clocks for Memory %chosen_mem% MHz (spaced across entire range):
echo ----------------------------------------------------------
set /a step=gpu_count / 15
if !step! equ 0 set step=1

set idx=1
set menu_idx=0

:loop_gpu
set /a menu_idx+=1
set /a actual_idx=idx

for %%k in (!actual_idx!) do (
    set "menu_gpu[!menu_idx!]=!gpu_val[%%k]!"
    echo [!menu_idx!] !gpu_val[%%k]! MHz
)

set /a idx+=step
if !idx! lss !gpu_count! goto loop_gpu

REM Print the minimum clock if it wasn't the last printed one
if !actual_idx! lss !gpu_count! (
    set /a menu_idx+=1
    for %%k in (!gpu_count!) do (
        set "menu_gpu[!menu_idx!]=!gpu_val[%%k]!"
        echo [!menu_idx!] !gpu_val[%%k]! MHz (Minimum supported)
    )
)
echo ----------------------------------------------------------
set /p gpu_choice="Select GPU Clock option (1-!menu_idx!), or enter a custom MHz value directly: "
if "%gpu_choice%"=="" (
    echo No selection made.
    goto cleanup_and_exit
)

set gpu_clock=
for %%i in (!gpu_choice!) do (
    if defined menu_gpu[%%i] (
        set gpu_clock=!menu_gpu[%%i]!
    )
)

if "!gpu_clock!"=="" (
    REM Assume user typed a custom MHz value
    set gpu_clock=!gpu_choice!
)

set memory_clock=%chosen_mem%
goto lock_clocks

:lock_clocks
REM Clean up files
if exist temp_mem.txt del temp_mem.txt
if exist temp_all.txt del temp_all.txt

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
exit /b 0

:cleanup_and_exit
if exist temp_mem.txt del temp_mem.txt
if exist temp_all.txt del temp_all.txt
pause
exit /b 1