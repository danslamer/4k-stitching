# Makefile for OpenCL Test Program on RK3576
# Target: Mali-G52 MC3 GPU (OpenCL 2.0)

# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lOpenCL -lm

# Source files
TARGET = cl_test
SRC = cl_test.c

# Default build
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $< $(LDFLAGS)

# Cross-compile for aarch64 (RK3576 is ARM64)
# Usage: make cross CROSS_COMPILE=/path/to/aarch64-toolchain-
cross: CC = $(CROSS_COMPILE)gcc
cross: $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $< $(LDFLAGS)

# Cross-compile with specific OpenCL headers
cross-ocl: CC = $(CROSS_COMPILE)gcc
cross-ocl: CFLAGS += -I$(OCL_INCLUDE_PATH)
cross-ocl: LDFLAGS += -L$(OCL_LIB_PATH)
cross-ocl: $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $< $(LDFLAGS)

# Clean build files
clean:
	rm -f $(TARGET) *.o

# Install to system (requires root)
install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

# Uninstall from system
uninstall:
	rm -f /usr/local/bin/$(TARGET)

# Help
help:
	@echo "OpenCL Test for RK3576 (Mali-G52) - Build Options"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build the test program (default)"
	@echo "  debug      - Build with debug symbols"
	@echo "  cross      - Cross-compile for aarch64"
	@echo "  clean      - Remove build files"
	@echo "  install    - Install to /usr/local/bin"
	@echo "  uninstall  - Remove from /usr/local/bin"
	@echo ""
	@echo "Cross-compilation example:"
	@echo "  make cross CROSS_COMPILE=aarch64-linux-gnu-"
	@echo ""
	@echo "Usage after building:"
	@echo "  ./$(TARGET) -h        Show help"
	@echo "  ./$(TARGET) -i        Show GPU info only"
	@echo "  ./$(TARGET) -b        Run benchmarks"
	@echo "  ./$(TARGET) -a        Run all tests"

.PHONY: all debug cross cross-ocl clean install uninstall help
