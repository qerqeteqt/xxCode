@echo off
rem ============================================================
rem  双击这个文件就能打开 xxCode 的网页界面。
rem
rem  它替你做了三件事：
rem    1. 切到脚本自己所在的目录（所以从资源管理器双击也行，
rem       不用先 cd）
rem    2. 找到对的那个 python —— 优先用当前激活的 conda 环境，
rem       没有就找 conda 的默认安装位置，再没有才退回 PATH
rem    3. 起服务并自动打开浏览器
rem
rem  想让它看别的项目，在最后一行加 --root：
rem      "%PYTHON%" main.py --web --root D:\pycharm\别的项目
rem ============================================================

chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON=%CONDA_PREFIX%\python.exe"
if not exist "%PYTHON%" set "PYTHON=%USERPROFILE%\.conda\envs\langgraph\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo 使用 Python: %PYTHON%
echo.

"%PYTHON%" main.py --web %*

rem 服务退出（Ctrl+C 或出错）后停一下，好看清楚报了什么
echo.
echo 服务已停止。
pause
