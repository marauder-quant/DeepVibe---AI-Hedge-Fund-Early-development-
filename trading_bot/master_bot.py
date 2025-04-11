import os
import sys
import subprocess
import time
import curses
from datetime import datetime
from threading import Thread, Lock
from collections import deque
import signal
import re

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError
from alpaca.trading.requests import GetPortfolioHistoryRequest

# --- Configuration ---
BOT_SCRIPTS = [
    "us_equity_bot.py",
    "chinese_equity_bot.py",
    "volatility_bot.py",
    "bonds_bot.py",
    "gold_bot.py",
]
HEADER_REFRESH_INTERVAL = 5  # seconds
MAX_LOG_LINES = 1000 # Maximum lines to keep in the log buffer
LOGS_DIR = "logs"  # Directory for storing log files

# Patterns to determine which log messages to show in minimal mode
# Only lines containing these patterns will be displayed
EXECUTION_PATTERNS = [
    r"order executed",       # Matches order execution confirmations
    r"trade",                # Any mentions of trades
    r"buy",                  # Buy trades
    r"sell",                 # Sell trades
    r"executed",             # Any executed actions
    r"filled",               # Order filled status
    r"position",             # Position changes
    r"purchase",             # Purchases
    r"execution",            # Any execution mentions
    r"transaction"           # Any transaction mentions
]
EXECUTION_PATTERN_REGEX = re.compile('|'.join(EXECUTION_PATTERNS), re.IGNORECASE)

# --- Globals ---
processes = []
log_lines = deque(maxlen=MAX_LOG_LINES)
log_lock = Lock()
animation_counter = 1  # Counter for bot status animation
keep_running = True
trading_client = None
initial_equity = 100000
log_files = {}  # Dictionary to store open log file handles

# --- Functions ---

def load_credentials():
    """Load Alpaca API credentials from .env file."""
    load_dotenv()
    api_key = os.environ.get('alpaca_paper_key')
    secret_key = os.environ.get('alpaca_paper_secret')
    if not api_key or not secret_key:
        raise ValueError("Alpaca API Key/Secret not found in .env file")
    return api_key, secret_key

def ensure_logs_directory(script_dir):
    """Ensure the logs directory exists."""
    logs_dir = os.path.join(script_dir, LOGS_DIR)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir

def get_log_file_path(logs_dir, script_name, is_error=False):
    """Get the log file path for a bot."""
    # Remove .py extension if present
    bot_name = script_name.replace(".py", "")
    # Add suffix for error logs
    suffix = ".err" if is_error else ""
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d')
    return os.path.join(logs_dir, f"{bot_name}{suffix}_{timestamp}.log")

def fetch_initial_equity(client):
    """Fetch portfolio history once to determine initial equity."""
    global initial_equity
    try:
        # Request portfolio history - adjust period/timeframe as needed
        # Using 10 years and daily timeframe as a broad starting point
        request_params = GetPortfolioHistoryRequest(
            period="10Y",       # Use string format based on user's library version
            timeframe="1D"      # Use string format based on user's library version
        )
        history = client.get_portfolio_history(request_params)
        if history.equity and len(history.equity) > 0:
            initial_equity = history.equity[0]
            # Don't log this in the UI to reduce noise
        else:
             # Only log critical errors
             with log_lock:
                log_lines.append("[ERROR MasterBot] Could not determine initial equity from portfolio history.")
    except APIError as e:
         with log_lock:
             log_lines.append(f"[ERROR MasterBot] API Error fetching portfolio history: {e}")
    except Exception as e:
        with log_lock:
             log_lines.append(f"[ERROR MasterBot] Error fetching portfolio history: {e}")

def get_account_summary(client):
    """Fetch account summary from Alpaca, including estimated All-Time PnL."""
    global initial_equity
    try:
        account = client.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity else 0

        # Calculate All-Time PnL based on fetched initial equity
        all_time_pnl = None
        all_time_pnl_pct = None
        if initial_equity is not None and initial_equity != 0:
            all_time_pnl = equity - initial_equity
            all_time_pnl_pct = (all_time_pnl / initial_equity * 100)
        elif initial_equity == 0:
             all_time_pnl = equity
             all_time_pnl_pct = float('inf') # Or handle as a special case

        return {
            "equity": equity,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "all_time_pnl": all_time_pnl,
            "all_time_pnl_pct": all_time_pnl_pct,
            "buying_power": float(account.buying_power),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except APIError as e:
        return {"error": f"API Error: {e}", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    except Exception as e:
        return {"error": f"Error fetching account: {e}", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

def read_output(pipe, script_name, is_error=False):
    """Read lines from a subprocess pipe, add to shared log and write to log file."""
    global log_files
    
    # Get script directory and ensure logs directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = ensure_logs_directory(script_dir)
    
    # Generate log file path and open file
    log_file_path = get_log_file_path(logs_dir, script_name, is_error)
    log_file = open(log_file_path, 'a', encoding='utf-8')
    log_files[script_name] = log_file
    
    try:
        for line in iter(pipe.readline, ''):
            if not keep_running:
                break
            
            # Timestamp and format the log line
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_line = f"[{timestamp}] {line.strip()}"
            
            # Write to log file (ALL output)
            log_file.write(log_line + '\n')
            log_file.flush()  # Ensure it's written immediately
                
            # Only add to UI log if it matches execution patterns or is from MasterBot
            line_str = line.strip()
            if EXECUTION_PATTERN_REGEX.search(line_str):
                with log_lock:
                    log_lines.append(f"[{script_name}] {line_str}")
            # Also log MasterBot startup messages - but ONLY for MasterBot
            elif script_name.startswith("MasterBot"):
                with log_lock:
                    log_lines.append(f"[{script_name}] {line_str}")
    except Exception as e:
         with log_lock:
            error_msg = f"[ERROR {script_name}] Pipe read error: {e}"
            log_lines.append(error_msg)
            log_file.write(error_msg + '\n')
    finally:
        pipe.close()
        log_file.close()
        if script_name in log_files:
            del log_files[script_name]

def start_bots(script_dir):
    """Start each bot script as a subprocess."""
    global processes
    python_executable = sys.executable # Use the same python interpreter
    
    # Ensure logs directory exists
    logs_dir = ensure_logs_directory(script_dir)
    
    # Log the startup event
    master_log_path = get_log_file_path(logs_dir, "MasterBot")
    with open(master_log_path, 'a', encoding='utf-8') as master_log:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        master_log.write(f"[{timestamp}] MasterBot starting all trading bots\n")

    for script in BOT_SCRIPTS:
        script_path = os.path.join(script_dir, script)
        if not os.path.exists(script_path):
            with log_lock:
                log_lines.append(f"[MasterBot] Error: Script not found - {script_path}")
            continue

        try:
            # Start the process
            # Pass PYTHONPATH to ensure imports work correctly if bots rely on relative project structure
            env = os.environ.copy()
            project_root = os.path.dirname(script_dir)
            env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

            process = subprocess.Popen(
                [python_executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                cwd=script_dir, # Run script from its directory
                env=env
            )
            processes.append(process)

            # Start threads to read stdout and stderr
            Thread(target=read_output, args=(process.stdout, script, False), daemon=True).start()
            Thread(target=read_output, args=(process.stderr, script, True), daemon=True).start()

            # Don't log every bot startup to reduce noise
            # Only log startup of first and last bot to confirm progress
            if script == BOT_SCRIPTS[0] or script == BOT_SCRIPTS[-1]:
                with log_lock:
                    log_lines.append(f"[MasterBot] Started {script}")

        except Exception as e:
            with log_lock:
                log_lines.append(f"[MasterBot] Failed to start {script}: {e}")

def stop_bots():
    """Signal and terminate all running bot processes."""
    global keep_running, log_files
    keep_running = False
    print("Stopping bots...") # Print outside curses

    # Log the shutdown in the master log
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = ensure_logs_directory(script_dir)
    master_log_path = get_log_file_path(logs_dir, "MasterBot")
    with open(master_log_path, 'a', encoding='utf-8') as master_log:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        master_log.write(f"[{timestamp}] MasterBot shutting down all trading bots\n")

    for p in processes:
        try:
            # Send SIGTERM first for graceful shutdown if bots handle it
            p.terminate()
        except ProcessLookupError:
            pass # Process already finished

    # Wait a bit for graceful termination
    time.sleep(2)

    # Force kill if still running
    for p in processes:
         try:
            if p.poll() is None: # Still running
                print(f"Force killing PID {p.pid}...")
                p.kill()
         except ProcessLookupError:
            pass
    
    # Close any remaining log files
    for file_handle in log_files.values():
        try:
            file_handle.close()
        except:
            pass
    log_files.clear()

def draw_ui(stdscr):
    """Main curses UI drawing loop."""
    global trading_client, animation_counter
    curses.curs_set(0) # Hide cursor
    stdscr.nodelay(True) # Non-blocking input
    stdscr.timeout(100) # Refresh timeout in ms

    # Colors (optional)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)

    header_win = None
    log_win = None
    last_header_update = 0
    account_summary = {"error": "Initializing..."}

    while keep_running:
        try:
            # Check terminal size
            max_y, max_x = stdscr.getmaxyx()
            header_height = 5  # Increased header height to include bot status line
            if max_y <= header_height:
                 stdscr.clear()
                 stdscr.addstr(0, 0, "Terminal too small!")
                 stdscr.refresh()
                 time.sleep(0.1)
                 continue

            # Recreate windows if size changed
            if header_win is None or log_win is None or header_win.getmaxyx() != (header_height, max_x) or log_win.getmaxyx() != (max_y - header_height, max_x):
                 if header_win: header_win.clear()
                 if log_win: log_win.clear()
                 header_win = curses.newwin(header_height, max_x, 0, 0)
                 log_win = curses.newwin(max_y - header_height, max_x, header_height, 0) # Adjusted y-position
                 log_win.scrollok(True) # Enable scrolling
                 log_win.idlok(True) # Enable hardware scrolling acceleration
                 # Redraw logs on resize
                 with log_lock:
                     log_win.clear()
                     start_idx = max(0, len(log_lines) - (max_y - header_height)) # Adjusted height
                     for i, line in enumerate(list(log_lines)[start_idx:]):
                          try:
                             log_win.addstr(i, 0, line[:max_x-1]) # Truncate long lines
                          except curses.error: pass # Ignore error if writing fails at bottom-right corner

            # --- Update Header ---
            current_time = time.time()
            if current_time - last_header_update > HEADER_REFRESH_INTERVAL:
                if trading_client:
                    account_summary = get_account_summary(trading_client)
                else:
                    account_summary = {"error": "Alpaca client not initialized"}
                last_header_update = current_time

                header_win.clear()
                header_win.bkgd(' ', curses.color_pair(3) | curses.A_BOLD)
                if "error" in account_summary:
                    header_win.addstr(0, 1, f"Status @ {account_summary.get('timestamp', 'N/A')}", curses.color_pair(4))
                    header_win.addstr(1, 1, f"Error: {account_summary['error']}", curses.color_pair(2))
                else:
                    equity = account_summary['equity']
                    daily_pnl = account_summary['daily_pnl']
                    daily_pnl_pct = account_summary['daily_pnl_pct']
                    all_time_pnl = account_summary.get('all_time_pnl')
                    all_time_pnl_pct = account_summary.get('all_time_pnl_pct')

                    daily_pnl_color = curses.color_pair(1) if daily_pnl >= 0 else curses.color_pair(2)
                    all_time_pnl_color = curses.color_pair(1)
                    if all_time_pnl is not None:
                        all_time_pnl_color = curses.color_pair(1) if all_time_pnl >= 0 else curses.color_pair(2)

                    header_win.addstr(0, 1, f"Portfolio @ {account_summary['timestamp']}", curses.color_pair(3) | curses.A_BOLD)
                    header_win.addstr(1, 1, f"Equity: ${equity:,.2f}", curses.A_BOLD)
                    header_win.addstr(2, 1, f"Day PnL:    ${daily_pnl:,.2f} ({daily_pnl_pct:.2f}%)    ", daily_pnl_color)

                    # Display All Time PnL
                    if all_time_pnl is not None and all_time_pnl_pct is not None:
                         header_win.addstr(3, 1, f"All Time PnL: ${all_time_pnl:,.2f} ({all_time_pnl_pct:.2f}%)    ", all_time_pnl_color)
                    else:
                        header_win.addstr(3, 1, "All Time PnL: N/A                   ", curses.color_pair(4))

            # Update animation counter for bot status (update every iteration)
            animation_counter = (animation_counter + 1) % 4
            animation_dots = "." * (animation_counter + 1) if animation_counter < 3 else "..."
            
            # Create bot status line
            running_bots = []
            for i, process in enumerate(processes):
                if i < len(BOT_SCRIPTS) and process.poll() is None:  # Process is still running
                    # Extract bot name without .py extension
                    bot_name = BOT_SCRIPTS[i].replace(".py", "")
                    running_bots.append(bot_name)
            
            if running_bots:
                status_text = "Bot Status: " + ", ".join(running_bots) + f" running{animation_dots}"
                header_win.addstr(4, 1, status_text[:max_x-2], curses.color_pair(1) | curses.A_BOLD)
            else:
                header_win.addstr(4, 1, f"Bot Status: No bots running{animation_dots}", curses.color_pair(2) | curses.A_BOLD)
            
            header_win.refresh()

            # --- Update Logs ---
            new_logs = []
            with log_lock:
                while len(log_lines) > 0 and len(new_logs) < 50: # Process logs in batches
                     try:
                         new_logs.append(log_lines.popleft())
                     except IndexError: # Should not happen with check but safety first
                         break

            if new_logs:
                for line in new_logs:
                    try:
                        # Ensure scrolling happens before adding the new line
                        log_win.scroll()
                        # Add new line at the bottom
                        log_win.addstr(max_y - header_height - 1, 0, line[:max_x-1]) # Truncate (Height adjusted)
                    except curses.error:
                        # Ignore error which can happen if window is small/resized quickly
                        pass
                log_win.refresh()
            else:
                # If no logs, small sleep to prevent busy-waiting
                 time.sleep(0.05)


            # Check for user input (e.g., 'q' to quit)
            key = stdscr.getch()
            if key == ord('q'):
                break

        except curses.error as e:
            # Handle curses errors, e.g., terminal resized too small
            stdscr.clear()
            stdscr.addstr(0, 0, f"Curses Error: {e}. Resize terminal or press 'q'.")
            stdscr.refresh()
            key = stdscr.getch() # Wait for keypress
            if key == ord('q'):
                break
        except KeyboardInterrupt:
            break # Exit loop on Ctrl+C

    stop_bots()

def signal_handler(sig, frame):
    """Handle signals like SIGINT (Ctrl+C)."""
    global keep_running
    print(f"Signal {sig} received, initiating shutdown...")
    keep_running = False
    # No need to call stop_bots here, it's called after the main loop exits

def main():
    global trading_client
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure logs directory exists
    ensure_logs_directory(script_dir)

    # Handle graceful shutdown signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        api_key, secret_key = load_credentials()
        trading_client = TradingClient(api_key, secret_key, paper=True) # Assuming paper trading
        with log_lock:
            log_lines.append("[MasterBot] Trading bots starting...")
        # Fetch initial equity after client is ready
        fetch_initial_equity(trading_client)

    except ValueError as e:
        print(f"Error initializing Alpaca client: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error initializing Alpaca client: {e}", file=sys.stderr)
        sys.exit(1)


    start_bots(script_dir)

    # Check if any bots actually started
    if not processes:
         print("No bot processes were started. Check script paths and permissions.", file=sys.stderr)
         sys.exit(1)

    try:
        curses.wrapper(draw_ui)
    finally:
        # Ensure cleanup happens even if curses wrapper fails
        if keep_running: # If wrapper exited abnormally, ensure stop is called
             stop_bots()
        print("Master Bot shut down.")

if __name__ == "__main__":
    main() 