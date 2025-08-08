#!/usr/bin/env python3
"""
Quick Analysis Runner for Game Reasoning Arena

A simple convenience script to run the most common analysis workflow
automatically without any manual steps.

Usage:
    python analysis/quick_analysis.py

This script will:
1. Find and merge all SQLite databases in the results/ directory
2. Run reasoning categorization and generate visualizations
3. Create comprehensive plots and charts
4. Output everything to the plots/ directory

For more advanced options, use run_full_analysis.py
"""

import sys
from pathlib import Path

# Add the analysis directory to the path
sys.path.append(str(Path(__file__).parent))

def main():
    """Run the most common analysis pipeline quickly."""
    try:
        # Import and run the full pipeline with default settings
        from run_full_analysis import AnalysisPipeline

        print("🚀 Starting Quick Game Reasoning Analysis...")
        print("   (For advanced options, use run_full_analysis.py)")
        print()

        # Initialize pipeline with sensible defaults
        pipeline = AnalysisPipeline(
            results_dir="results",
            plots_dir="plots",
            verbose=True,
            skip_existing=False
        )

        # Run the complete pipeline
        success = pipeline.run_full_pipeline()

        if success:
            print()
            print("✅ Analysis completed successfully!")
            print("📊 Check the plots/ directory for all generated visualizations")
            print("📋 See analysis_summary_report.json for detailed results")
        else:
            print()
            print("❌ Analysis completed with some errors")
            print("📋 Check the console output above for details")
            sys.exit(1)

    except ImportError as e:
        print(f"❌ Error importing analysis modules: {e}")
        print("Make sure you're running this from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
