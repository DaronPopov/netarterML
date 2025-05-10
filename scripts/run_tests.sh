#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Function to show help
show_help() {
    echo "Usage: $0 [options] [test_paths...]"
    echo
    echo "Options:"
    echo "  --unit        Run only unit tests"
    echo "  --integration Run only integration tests"
    echo "  --coverage    Run tests with coverage report"
    echo "  --verbose     Run tests in verbose mode"
    echo "  --help        Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --unit"
    echo "  $0 --integration --coverage"
    echo "  $0 tests/test_specific_file.py"
}

# Default options
PYTEST_ARGS=()
COVERAGE=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --unit)
            PYTEST_ARGS+=("tests/unit")
            shift
            ;;
        --integration)
            PYTEST_ARGS+=("tests/integration")
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            PYTEST_ARGS+=("-v")
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# If no specific test paths provided, run all tests
if [ ${#PYTEST_ARGS[@]} -eq 0 ]; then
    PYTEST_ARGS=("tests")
fi

# Add coverage if requested
if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS=("--cov=OPENtransformer" "--cov-report=term-missing" "${PYTEST_ARGS[@]}")
fi

# Run the tests
cd "$PROJECT_ROOT" && python -m pytest "${PYTEST_ARGS[@]}" 