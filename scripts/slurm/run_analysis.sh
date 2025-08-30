#!/bin/bash

# Game Reasoning Arena - Analysis Runner Script
#
# This script makes it easy to run the complete analysis pipeline
# without having to remember Python commands or paths.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if we're in the right directory
check_directory() {
    if [[ ! -f "README.md" ]] || [[ ! -d "analysis" ]]; then
        print_error "This script must be run from the game_reasoning_arena project root directory"
        exit 1
    fi
}

# Function to check if results directory exists
check_results() {
    if [[ ! -d "results" ]]; then
        print_error "No 'results' directory found. Run some games first to generate data."
        exit 1
    fi

    if [[ -z "$(find results -name '*.db' -print -quit)" ]]; then
        print_error "No .db files found in results/ directory. Run some games first to generate data."
        exit 1
    fi
}

# Function to run the analysis
run_analysis() {
    print_info "Starting Game Reasoning Arena Analysis Pipeline..."
    echo

    # Check if Python 3 is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not found in PATH"
        exit 1
    fi

    # Run the analysis
    if python3 analysis/quick_analysis.py; then
        echo
        print_success "Analysis completed successfully!"
        print_info "Results are available in the plots/ directory"

        # List some of the generated files
        if [[ -d "plots" ]]; then
            echo
            print_info "Generated files include:"
            find plots -name "*.png" -type f | head -5 | while read file; do
                echo "  📊 $file"
            done

            total_plots=$(find plots -name "*.png" -type f | wc -l)
            if [[ $total_plots -gt 5 ]]; then
                echo "  ... and $((total_plots - 5)) more plot files"
            fi
        fi
    else
        echo
        print_error "Analysis failed. Check console output above for details."
        exit 1
    fi
}

# Main script
main() {
    echo "🎮 Game Reasoning Arena - Analysis Runner"
    echo "========================================"
    echo

    # Check if we're in the right place
    check_directory

    # Check if we have data to analyze
    check_results

    # Run the analysis
    run_analysis

    echo
    print_success "All done! 🎉"
}

# Help function
show_help() {
    echo "Game Reasoning Arena - Analysis Runner"
    echo
    echo "Usage: $0 [OPTION]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -q, --quiet    Run analysis in quiet mode"
    echo "  -f, --full     Run full analysis with all options"
    echo
    echo "This script will:"
    echo "  1. Check for SQLite database files in results/"
    echo "  2. Merge all databases into a consolidated CSV"
    echo "  3. Run reasoning pattern analysis"
    echo "  4. Generate comprehensive visualizations"
    echo "  5. Create summary reports"
    echo
    echo "Output will be saved to the plots/ directory."
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -q|--quiet)
        print_info "Running analysis in quiet mode..."
        python3 analysis/run_full_analysis.py --quiet
        ;;
    -f|--full)
        print_info "Running full analysis with all options..."
        python3 analysis/run_full_analysis.py
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
