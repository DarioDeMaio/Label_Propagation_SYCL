@echo off
setlocal enabledelayedexpansion

set NUM_VERTICES=8000
set NUM_HYPEREDGES=50000
set PROBABILITY=0.01
set OUTPUT_FILE=output.txt

rem Cancella il file se esiste
if exist %OUTPUT_FILE% del %OUTPUT_FILE%

for /L %%i in (1,1,10) do (
    echo Esecuzione %%i... >> %OUTPUT_FILE%
    label_prop.exe %NUM_VERTICES% %NUM_HYPEREDGES% %PROBABILITY% >> %OUTPUT_FILE%
    echo ---------------------------- >> %OUTPUT_FILE%
    echo. >> %OUTPUT_FILE%
)

echo Completato. I risultati sono in %OUTPUT_FILE%
pause
