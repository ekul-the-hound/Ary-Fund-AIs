@echo off
echo.
echo  Hedge Fund AI - Activating environment...
cd /d "%~dp0"
call "%~dp0hedgefund_ai\Scripts\activate.bat"
echo  Done. You are now in (hedgefund_ai) at %CD%
echo  Run: python main.py
echo.
cmd /k