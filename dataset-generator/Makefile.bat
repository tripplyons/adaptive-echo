@echo off
echo "Wiping Build Directory..."
IF EXIST build rmdir /s /q build

echo "Generating Build Directory And Compiling.."
mkdir build
cd build
cmake ..
cmake --build .
cd ..