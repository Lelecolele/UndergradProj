@echo off

call D:\Leko\Anaconda\Scripts\activate.bat
cd /d D:\Leko\final\bladder_ai_app
call conda activate multi_task
python app.py

pause
