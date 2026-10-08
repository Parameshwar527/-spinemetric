@echo off
cd /d C:\SpineMetric\backend
start "" powershell -NoProfile -Command "for($i=0;$i -lt 90;$i++){try{Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing -TimeoutSec 1 | Out-Null; Start-Process http://127.0.0.1:8000; break}catch{Start-Sleep 1}}"
python -m uvicorn app:app --port 8000
pause