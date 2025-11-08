@echo off
set PROJECT_NAME=AdaptiveEchoDatasetGenerator
set BUILD_DIR=build
set INSTALL_DIR=%CD%\install

echo "Wiping Build Directory..."
IF EXIST build rmdir /s /q %BUILD_DIR%

echo "Generating Build Directory..."
mkdir %BUILD_DIR%
cd %BUILD_DIR%
cmake .. -G "Ninja" -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_BUILD_TYPE=Release -DCMAKE_WIN32_LONG_PATHS=ON
echo Building project...
cmake --build .
echo "Done..."