import os
import sys
import warnings

# Filter out weight_norm FutureWarning deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
import multiprocessing as mp

from const import ROOT_PATH, UPLOAD_DIR, TMP_DIR, LOG_FILE, get_version, get_edition

import asyncio
import logging
import argparse
import signal
import asyncio
from datetime import datetime
from Exceptions import setup_event_loop

from downloader.ModelManager import ModelManager
from settings import get_settings
from webserver.server import WebServer

# Add the project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# NOTE: This is required to fix current working directory on macOS
os.chdir(ROOT_PATH)

# NOTE: This is required to avoid recursive process call bug for macOS
mp.freeze_support()

# Initialize settings and logger at module level
settings = get_settings()
logger = logging.getLogger(__name__)

def setup_logging(log_level: str = 'info'):
    """Configure logging for the application."""
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level.upper())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s %(levelname)-8s [%(module)s] %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), stream_handler]
    )
    return logging.getLogger(__name__)

def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Run the voice changer server.")
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="info", 
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level"
    )
    parser.add_argument(
        "--launch-browser",
        action="store_true",
        help="Open the web interface in the default browser on startup"
    )
    return parser

def _get_lan_ip() -> str | None:
    """Return the machine's primary LAN IPv4 address, or None if not found."""
    import socket as _sock
    try:
        with _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return None


def _read_key(timeout_remaining: float) -> str | None:
    """Non-blocking single-character read. Returns the char or None."""
    try:
        if not sys.stdin.isatty():
            return None
        if sys.platform == 'win32':
            import msvcrt
            if msvcrt.kbhit():
                return msvcrt.getch().decode('utf-8', errors='ignore')
        else:
            import select
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                line = sys.stdin.readline()
                if not line:  # EOF reached
                    return None
                return line.strip()[:1]
    except Exception:
        pass
    return None


def _run_countdown(prompt: str, valid: set, timeout: int) -> str | None:
    """Show a countdown prompt and return the first valid key pressed, or None on timeout."""
    import time
    start = time.time()
    choice = None
    while time.time() - start < timeout:
        remaining = int(timeout - (time.time() - start))
        sys.stdout.write(f"\r  {prompt} ({remaining}s)  ")
        sys.stdout.flush()
        ch = _read_key(timeout - (time.time() - start))
        if ch and ch in valid:
            choice = ch
            print(f"\n\n  Selected: [{choice}]")
            break
        import time as _t
        _t.sleep(0.05)
    return choice


def show_pre_boot_menu(timeout: int = 8) -> tuple[str, str]:
    """
    Shows a two-step 'Pre-Boot BIOS' style menu.

    Step 1 — Protocol:  [1] HTTP  [2] HTTPS
    Step 2 — Bind addr: [1] 127.0.0.1  [2] <LAN IP>  [3] 0.0.0.0

    Returns (protocol, host) where protocol is 'http' or 'https'
    and host is the chosen bind address string.
    """
    import time

    lan_ip = _get_lan_ip()

    # ── Step 1: Protocol ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("              AURA-ECHO  PRE-BOOT  BIOS  MENU              ")
    print("=" * 60)
    print("  STEP 1 of 2  —  Select network protocol")
    print()
    print("  [1]  HTTP   — standard, unencrypted  (localhost only)")
    print("  [2]  HTTPS  — self-signed TLS         (required for LAN)")
    print("-" * 60)
    print("  Auto-boot: HTTP + localhost")
    print("=" * 60)

    ch = _run_countdown("Auto-booting HTTP", {'1', '2'}, timeout)
    if ch is None:
        print("\n\n  Timeout — defaulting to HTTP + localhost.")
        print("=" * 60 + "\n")
        return 'http', '127.0.0.1'

    protocol = 'https' if ch == '2' else 'http'

    # ── Step 2: Bind address ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("  STEP 2 of 2  —  Select bind address")
    print()
    print("  [1]  127.0.0.1     — localhost only   (this machine)")
    if lan_ip:
        print(f"  [2]  {lan_ip:<15}— LAN / local network")
        print("  [3]  0.0.0.0        — all interfaces   (LAN + external)")
        valid_addr = {'1', '2', '3'}
    else:
        print("  [2]  0.0.0.0        — all interfaces   (LAN + external)")
        valid_addr = {'1', '2'}
    print("-" * 60)
    if protocol == 'https':
        print("  NOTE: HTTPS is required for microphone access over LAN.")
    else:
        print("  NOTE: HTTP + non-localhost will trigger a browser warning.")
    print("  Auto-select: 127.0.0.1")
    print("=" * 60)

    ch2 = _run_countdown("Auto-selecting localhost", valid_addr, timeout)

    if ch2 is None or ch2 == '1':
        host = '127.0.0.1'
    elif ch2 == '2':
        host = lan_ip if lan_ip else '0.0.0.0'
    else:
        host = '0.0.0.0'

    # Enforce HTTPS when binding to a non-localhost address
    if host not in ('127.0.0.1', 'localhost') and protocol == 'http':
        print("\n  [AUTO] Non-localhost address selected — upgrading to HTTPS")
        print("         (browsers require a secure context for microphone access)")
        protocol = 'https'

    print(f"\n  Boot configuration: {protocol.upper()} on {host}:{settings.port}")
    print("=" * 60 + "\n")
    return protocol, host


async def main():
    """Main entry point for the application."""
    # Show BIOS selection menu before starting the server
    boot_mode, boot_host = show_pre_boot_menu(timeout=8)

    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.log_level)

    # Apply selected boot mode to settings
    settings.host = boot_host
    if boot_mode == 'https':
        settings.ssl_enabled = True
        logger.info(f"Selected Boot Mode: HTTPS (Encrypted SSL/TLS) on {boot_host}")
    else:
        settings.ssl_enabled = False
        logger.info(f"Selected Boot Mode: HTTP (Unencrypted) on {boot_host}")
    
    logger.info(f"Python: {sys.version}")
    logger.info(f"Voice changer version: {get_version()} {get_edition()}")
    
    # Create necessary directories
    os.makedirs(settings.model_dir, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    
    logger.info(f"Server settings: {settings}")
    
    # Initialize and start the web server
    server = WebServer(
        host=settings.host,
        port=settings.port,
        log_level=args.log_level
    )
    
    # Start checking and downloading mandatory models in the background asynchronously
    logger.info("Initializing mandatory models check in the background...")
    asyncio.create_task(ModelManager.check_and_download_mandatory_models())
    
    # Start the server
    await server.start(
        launch_browser=args.launch_browser,
        ssl_keyfile=settings.ssl_keyfile,
        ssl_certfile=settings.ssl_certfile,
        ssl_self_signed=settings.ssl_enabled and not (settings.ssl_keyfile and settings.ssl_certfile)
    )



async def shutdown(signal, loop, server=None):
    """Cleanup tasks tied to the service's shutdown."""
    signal_name = signal.name if hasattr(signal, 'name') else str(signal)
    logger.info(f"Received exit signal {signal_name}...")
    
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    
    if server:
        await server.shutdown()
        
    loop.stop()

def handle_exception(loop, context):
    """Handle uncaught exceptions in the event loop."""
    msg = context.get("exception", context["message"])
    logger.error(f"Caught exception: {msg}")
    logger.error("Shutting down...")
    asyncio.create_task(shutdown(signal.SIGTERM, loop))

if __name__ == "__main__":
    server = None
    try:
        # Initialize settings and logger at the module level
        settings = get_settings()
        logger = setup_logging('info')
        
        # Set up event loop with connection reset handling
        loop = setup_event_loop()
        
        # Set up signal handlers (Windows-compatible)
        if os.name == 'nt':  # Windows
            signals = [signal.SIGINT, signal.SIGTERM]
            # On Windows, only these signals are available
            for sig in signals:
                signal.signal(sig, lambda s, _: asyncio.create_task(shutdown(s, loop, server)))
        else:  # Unix
            signals = [signal.SIGHUP, signal.SIGTERM, signal.SIGINT]
            for sig in signals:
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(shutdown(s, loop, server))
                )
        
        # Set exception handler
        loop.set_exception_handler(handle_exception)
        
        # Run the application with a timeout
        server = loop.run_until_complete(main())
        
        # Keep the application running until interrupted
        loop.run_forever()
        
    except KeyboardInterrupt:
        logger.info("\nShutdown requested. Cleaning up...")
    except Exception as e:
        if 'logger' in globals():
            logger.exception("An error occurred while running the server")
        else:
            print(f"Critical error: {str(e)}")
        raise e
    finally:
        # Clean up the event loop
        if 'loop' in locals():
            tasks = asyncio.all_tasks(loop)
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.close()
            logger.info("Shutdown complete")
