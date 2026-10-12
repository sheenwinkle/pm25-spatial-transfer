@echo off
rem Weighting check for the fixed-window analysis (thesis Section 5.6), started 12 Oct 2026.
rem Re-runs the 200 window/seed/split fits with bench_window_sensitivity_sw.py, which also fits
rem NNW[LDF] with the NNW weights passed as sample weights, using the same LDF codes and weights.
rem All other columns should reproduce audit\round8_20261009\window_results.
setlocal
set "ROOT=%~dp0..\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "EXP=%ROOT%\benchmark_replication_20261008\benchmark_experiment"
set "PY=%ROOT%\..\..\tmp\pm25-round8-env\Scripts\python.exe"
for %%I in ("%PY%") do set "PY=%%~fI"
set "OUT=%~dp0window_results_sw"
if not exist "%OUT%" mkdir "%OUT%"
echo started %DATE% %TIME% processors=%NUMBER_OF_PROCESSORS%> "%OUT%\launch.txt"
echo python=%PY%>> "%OUT%\launch.txt"
if not exist "%PY%" echo MISSING PYTHON>> "%OUT%\launch.txt"
if %NUMBER_OF_PROCESSORS% GEQ 8 goto four
start "weighting 00-10" /D "%EXP%" /MIN cmd /c ""%PY%" bench_window_sensitivity_sw.py --split-start 0 --split-stop 10 --out "%OUT%" > "%OUT%\batch_00_10.log" 2>&1"
start "weighting 10-20" /D "%EXP%" /MIN cmd /c ""%PY%" bench_window_sensitivity_sw.py --split-start 10 --split-stop 20 --out "%OUT%" > "%OUT%\batch_10_20.log" 2>&1"
goto done
:four
start "weighting 00-05" /D "%EXP%" /MIN cmd /c ""%PY%" bench_window_sensitivity_sw.py --split-start 0 --split-stop 5 --out "%OUT%" > "%OUT%\batch_00_05.log" 2>&1"
start "weighting 05-10" /D "%EXP%" /MIN cmd /c ""%PY%" bench_window_sensitivity_sw.py --split-start 5 --split-stop 10 --out "%OUT%" > "%OUT%\batch_05_10.log" 2>&1"
start "weighting 10-15" /D "%EXP%" /MIN cmd /c ""%PY%" bench_window_sensitivity_sw.py --split-start 10 --split-stop 15 --out "%OUT%" > "%OUT%\batch_10_15.log" 2>&1"
start "weighting 15-20" /D "%EXP%" /MIN cmd /c ""%PY%" bench_window_sensitivity_sw.py --split-start 15 --split-stop 20 --out "%OUT%" > "%OUT%\batch_15_20.log" 2>&1"
:done
endlocal
