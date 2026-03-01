@echo off
setlocal
cd /d D:\Veg

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo ================================================== >> D:\Veg\logs\crawl.log
echo [%date% %time%] START crawl_recent >> D:\Veg\logs\crawl.log

D:\DevelopTools\Anaconda\envs\veg_env\python.exe D:\Veg\manage.py crawl_recent --days 7 --overlap 7 >> D:\Veg\logs\crawl.log 2>&1

echo [%date% %time%] END crawl_recent >> D:\Veg\logs\crawl.log
