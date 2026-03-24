#!/bin/bash
#
# run_test.sh - OpenCL Test Script for MYD-LR3576
#
# This script compiles and runs the OpenCL test program
# on the RK3576 development board with Mali-G52 GPU.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

# Print header
print_header() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     OpenCL Test Script for MYD-LR3576                  ║${NC}"
    echo -e "${BLUE}║               Mali-G52 MC3 GPU                         ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Check if running on RK3576
check_platform() {
    print_msg "Checking platform..." "$YELLOW"
    
    if [ -f /proc/device-tree/compatible ]; then
        compatible=$(cat /proc/device-tree/compatible 2>/dev/null | tr '\0' ' ')
        if echo "$compatible" | grep -q "rk3576"; then
            print_msg "✓ Detected RK3576 platform" "$GREEN"
            return 0
        fi
    fi
    
    if [ -f /proc/cpuinfo ]; then
        if grep -q "rk3576\|RK3576" /proc/cpuinfo 2>/dev/null; then
            print_msg "✓ Detected RK3576 platform" "$GREEN"
            return 0
        fi
    fi
    
    print_msg "⚠ Platform detection failed, continuing anyway..." "$YELLOW"
    return 0
}

# Check OpenCL dependencies
check_deps() {
    print_msg "Checking OpenCL dependencies..." "$YELLOW"
    
    local missing=()
    
    # Check for OpenCL library
    local ocl_found=0
    
    # Check common OpenCL library locations
    for lib in /usr/lib/libOpenCL.so /usr/lib/*/libOpenCL.so \
               /usr/lib/libmali.so /usr/lib/*/libmali.so \
               /usr/lib/aarch64-linux-gnu/libOpenCL.so \
               /usr/lib/aarch64-linux-gnu/libmali.so; do
        if [ -f "$lib" ]; then
            print_msg "✓ Found OpenCL library: $lib" "$GREEN"
            ocl_found=1
        fi
    done
    
    # Check for ICD files
    if ls /etc/OpenCL/vendors/*.icd >/dev/null 2>&1; then
        print_msg "✓ Found OpenCL ICD configuration" "$GREEN"
        for icd in /etc/OpenCL/vendors/*.icd; do
            print_msg "  - $(basename $icd)" "$CYAN"
        done
        ocl_found=1
    fi
    
    if [ $ocl_found -eq 0 ]; then
        print_msg "✗ OpenCL library not found!" "$RED"
        missing+=("libOpenCL or Mali GPU driver")
    fi
    
    # Check for clinfo utility
    if command -v clinfo >/dev/null 2>&1; then
        print_msg "✓ clinfo utility available" "$GREEN"
    else
        print_msg "⚠ clinfo not installed (optional)" "$YELLOW"
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo ""
        print_msg "Missing dependencies: ${missing[*]}" "$RED"
        echo ""
        echo "Installation instructions:"
        echo ""
        echo "  For RK3576 with Mali GPU:"
        echo "    1. Install Mali GPU driver from vendor SDK"
        echo "    2. Or install OpenCL ICD:"
        echo "       sudo apt install ocl-icd-opencl-dev"
        echo ""
        echo "  For development (headers):"
        echo "       sudo apt install opencl-headers"
        echo ""
        return 1
    fi
    
    print_msg "✓ All dependencies satisfied" "$GREEN"
    return 0
}

# Compile the test program
compile_test() {
    print_msg "Compiling OpenCL test program..." "$YELLOW"
    
    cd "$SCRIPT_DIR"
    
    # Clean previous build
    make clean >/dev/null 2>&1 || true
    
    # Compile
    if make 2>&1; then
        print_msg "✓ Build successful!" "$GREEN"
        return 0
    else
        print_msg "✗ Build failed!" "$RED"
        echo ""
        echo "Make sure you have installed:"
        echo "  sudo apt install build-essential opencl-headers"
        return 1
    fi
}

# Run the test
run_test() {
    print_msg "Running OpenCL test..." "$YELLOW"
    echo ""
    
    cd "$SCRIPT_DIR"
    
    if [ ! -x ./cl_test ]; then
        print_msg "✗ Test program not found or not executable" "$RED"
        return 1
    fi
    
    # Set library path if needed
    if [ -d /usr/lib/mali ]; then
        export LD_LIBRARY_PATH=/usr/lib/mali:$LD_LIBRARY_PATH
    fi
    
    # Run with provided arguments or default
    ./cl_test "$@"
    local ret=$?
    
    echo ""
    if [ $ret -eq 0 ]; then
        print_msg "✓ Test completed successfully!" "$GREEN"
    else
        print_msg "✗ Test failed with code $ret" "$RED"
    fi
    
    return $ret
}

# Show GPU info only
show_info() {
    cd "$SCRIPT_DIR"
    
    if [ -x ./cl_test ]; then
        ./cl_test -i
    else
        print_msg "Please compile first: ./run_test.sh compile" "$YELLOW"
    fi
}

# Quick test (info only)
quick_test() {
    check_platform
    check_deps || true
    compile_test || exit 1
    ./cl_test -i
}

# Full benchmark
full_benchmark() {
    check_platform
    check_deps || exit 1
    compile_test || exit 1
    ./cl_test -a
}

# Print usage
print_usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  check     Check platform and OpenCL dependencies"
    echo "  compile   Compile the test program"
    echo "  run       Run the test program"
    echo "  info      Show GPU info only"
    echo "  quick     Quick test (info only)"
    echo "  bench     Full benchmark test"
    echo "  clean     Clean build files"
    echo "  help      Show this help"
    echo ""
    echo "Run options (passed to cl_test):"
    echo "  -i        Show platform/device info"
    echo "  -b        Run all benchmarks"
    echo "  -v        Vector operations benchmark"
    echo "  -m        Matrix multiplication benchmark"
    echo "  -M        Memory bandwidth benchmark"
    echo "  -a        Run all tests"
    echo ""
    echo "Examples:"
    echo "  $0 quick                  # Quick info test"
    echo "  $0 bench                  # Full benchmark"
    echo "  $0 run -i                 # Show GPU info"
    echo "  $0 run -v -m              # Vector + matrix benchmarks"
}

# Main
main() {
    print_header
    
    local cmd="${1:-quick}"
    shift || true
    
    case "$cmd" in
        check)
            check_platform
            check_deps
            ;;
        compile)
            compile_test
            ;;
        run)
            run_test "$@"
            ;;
        info)
            show_info
            ;;
        quick)
            quick_test
            ;;
        bench|benchmark)
            full_benchmark
            ;;
        clean)
            make -C "$SCRIPT_DIR" clean
            print_msg "✓ Cleaned" "$GREEN"
            ;;
        help|--help|-h)
            print_usage
            ;;
        *)
            print_msg "Unknown command: $cmd" "$RED"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
