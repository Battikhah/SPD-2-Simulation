@echo off
rem Host-only parity build with Visual Studio 2022 (MSVC). Run from the repository root:
rem   firmware\logmel\test\build_msvc.cmd
rem Produces outputs\logmel-parity\parity_test.exe (float, as on the ESP32) and
rem parity_test_double.exe (double arithmetic, for diagnosing float32 error only).
setlocal
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set VS=%%i
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not exist outputs\logmel-parity mkdir outputs\logmel-parity
cl /nologo /std:c11 /W4 /WX /O2 /fp:precise firmware\logmel\logmel.c firmware\logmel\test\parity_test.c /Fo:outputs\logmel-parity\ /Fe:outputs\logmel-parity\parity_test.exe || exit /b 1
cl /nologo /std:c11 /W4 /WX /O2 /fp:precise /DLOGMEL_USE_LOGF firmware\logmel\logmel.c firmware\logmel\test\parity_test.c /Fo:outputs\logmel-parity\ /Fe:outputs\logmel-parity\parity_test_logf.exe || exit /b 1
cl /nologo /std:c11 /W4 /O2 /fp:precise /DLOGMEL_REAL=double firmware\logmel\logmel.c firmware\logmel\test\parity_test.c /Fo:outputs\logmel-parity\ /Fe:outputs\logmel-parity\parity_test_double.exe || exit /b 1
