@echo off
rem 重新打包单文件 exe，输出到 dist\LetsMinesweeperBot.exe
cd /d "%~dp0"
python -m PyInstaller --noconsole --onefile --uac-admin --clean --name LetsMinesweeperBot app.py
echo.
echo 完成：%~dp0dist\LetsMinesweeperBot.exe
pause
