@echo off
REM Run compute-sanitizer over the correctness binary, save logs to results/sanitizer/.
REM Usage: scripts\run_sanitizer.bat
setlocal
if not exist results\sanitizer mkdir results\sanitizer
where compute-sanitizer >nul 2>nul
if errorlevel 1 (
  echo compute-sanitizer NOT FOUND. Install the CUDA toolkit, then re-run.
  echo Marking sanitizer evidence NOT RUN.
  exit /b 0
)
for %%T in (memcheck racecheck synccheck initcheck) do (
  echo [sanitizer] tool=%%T
  compute-sanitizer --tool %%T build\kern_correctness.exe --seed 1234 > results\sanitizer\%%T.log 2>&1
  if errorlevel 1 (
    echo [sanitizer] %%T FAILED - see results\sanitizer\%%T.log
  ) else (
    echo [sanitizer] %%T clean
  )
)
echo Done. Commit results\sanitizer\*.log.
