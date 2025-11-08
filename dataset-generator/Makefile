all: build

build:
	@echo "Wiping Build Directory..."
	@if [ -d build ]; then rm -rf build; fi
	@echo "Generating Build Directory And Compiling.."
	@mkdir build
	@cd build && cmake ..
	@cd build && cmake --build .

.PHONY: all build