# Bot Logs Directory

This directory contains log files for each trading bot managed by the master_bot.py script.

## Log File Format

Log files follow this naming convention:
- Standard output: `botname_YYYYMMDD.log` 
- Error output: `botname.err_YYYYMMDD.log`

Where:
- `botname` is the name of the bot script without the .py extension
- `YYYYMMDD` is the date when the log was created

## Log File Content

Each log file contains all terminal output from its corresponding bot, with timestamps added. Each line follows this format:

```
[YYYY-MM-DD HH:MM:SS] Original log message
```

## MasterBot Logs

The MasterBot has its own log file that records startup, shutdown, and other system-level events.

## Log Rotation

Logs are created with daily timestamps. Each day will generate new log files, making it easy to manage historical logs. 