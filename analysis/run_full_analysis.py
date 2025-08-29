#!/usr/bin/env python3
"""
Full Analysis Pipeline for Game Reasoning Arena

This script orchestrates the complete analysis workflow by running all analysis
scripts in the correct order automatically. No more manual script execution!

Usage:

    python3 analysis/run_full_analysis.py [options]

Options:
    --results-dir DIR     Directory containing SQLite database files
                        (default: results)
    --plots-dir DIR       Directory for output plots and visualizations
                        (default: plots)
    --quiet               Run in quiet mode (less verbose output)
    --skip-existing       Skip analysis steps if output files already exist

Examples:
    python3 analysis/run_full_analysis.py
    python3 analysis/run_full_analysis.py --results-dir my_results \
        --plots-dir my_plots
    python3 analysis/run_full_analysis.py --quiet
    python3 analysis/run_full_analysis.py --skip-existing

Features:
    - Automatic data processing and merging
    - Comprehensive reasoning analysis
    - All visualizations generation
    - Configurable output directories
    - Progress tracking and logging
    - Error handling and recovery
"""


import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np


def json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert both keys and values
        return {
            json_serializable(k): json_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(json_serializable(v) for v in obj)
    else:
        return obj


# Add the parent directory to sys.path to import analysis modules
sys.path.append(str(Path(__file__).parent))

# Import analysis modules
try:
    from post_game_processing import (merge_sqlite_logs,
                                      compute_summary_statistics)
    from reasoning_analysis import LLMReasoningAnalyzer
except ImportError as e:
    print(f"Error importing analysis modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


class AnalysisPipeline:
    """Orchestrates the complete analysis pipeline."""

    def __init__(self,
                 results_dir: str = "results",
                 plots_dir: str = "plots",
                 verbose: bool = True,
                 skip_existing: bool = False,
                 game_filter: Optional[str] = None,
                 model_filter: Optional[str] = None):
        """
        Initialize the analysis pipeline.

        Args:
            results_dir: Directory containing SQLite database files
            plots_dir: Directory for output plots and visualizations
            verbose: Enable verbose logging
            skip_existing: Skip steps if output files already exist
            game_filter: Filter analysis for specific game (e.g., 'hex')
            model_filter: Filter analysis for specific model
        """
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.verbose = verbose
        self.skip_existing = skip_existing
        self.game_filter = game_filter
        self.model_filter = model_filter

        # Setup logging
        log_level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Create output directories
        self.plots_dir.mkdir(exist_ok=True)

        # Pipeline results tracking
        self.pipeline_results = {
            "start_time": None,
            "end_time": None,
            "steps_completed": [],
            "steps_failed": [],
            "files_generated": [],
            "summary": {}
        }

    def log_step(self, step_name: str, status: str, details: str = ""):
        """Log pipeline step progress."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info("[%s] %s: %s", step_name, status, details)

        if status == "COMPLETED":
            self.pipeline_results["steps_completed"].append({
                "step": step_name,
                "timestamp": timestamp,
                "details": details
            })
        elif status == "FAILED":
            self.pipeline_results["steps_failed"].append({
                "step": step_name,
                "timestamp": timestamp,
                "error": details
            })

    def step_1_merge_databases(self) -> Optional[str]:
        """Step 1: Merge all SQLite databases into a consolidated CSV."""
        step_name = "Database Merging"
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 1: %s", step_name)
        self.logger.info("="*60)

        try:
            # Check if results directory exists and has .db files
            if not self.results_dir.exists():
                raise FileNotFoundError(
                    f"Results directory {self.results_dir} not found"
                )

            db_files = list(self.results_dir.glob("*.db"))
            # Filter out human player databases
            db_files = [db for db in db_files
                        if not db.stem.startswith("human")]
            if not db_files:
                raise FileNotFoundError(
                    f"No .db files found in {self.results_dir}"
                )

            self.logger.info("Found %d database files to merge "
                             "(excluding human player data)", len(db_files))

            # Merge databases
            merged_df = merge_sqlite_logs(str(self.results_dir))

            if merged_df.empty:
                raise ValueError(
                    "Merged dataframe is empty - no data to analyze"
                )

            # Find the latest merged file
            merged_files = list(self.results_dir.glob("merged_logs_*.csv"))
            if not merged_files:
                raise FileNotFoundError("No merged CSV file was created")

            latest_merged = max(merged_files, key=lambda x: x.stat().st_mtime)

            # Compute summary statistics
            summary_stats = compute_summary_statistics(merged_df)

            self.pipeline_results["files_generated"].append(str(latest_merged))
            self.pipeline_results["summary"]["database_merge"] = {
                "total_records": len(merged_df),
                "db_files_processed": len(db_files),
                "merged_file": str(latest_merged),
                "summary_stats": summary_stats
            }

            self.log_step(
                step_name, "COMPLETED",
                f"Merged {len(db_files)} databases into {latest_merged} "
                f"({len(merged_df)} records)"
            )

            return str(latest_merged)

        except Exception as e:
            self.log_step(step_name, "FAILED", str(e))
            return None

    def _apply_filters(self, analyzer) -> bool:
        """Apply game and model filters to the analyzer data."""
        if not self.game_filter and not self.model_filter:
            return True  # No filtering needed

        original_size = len(analyzer.df)

        # Apply game filter
        if self.game_filter:
            self.logger.info("Filtering for game: %s", self.game_filter)
            analyzer.df = analyzer.df[
                analyzer.df['game_name'] == self.game_filter
            ]

        # Apply model filter
        if self.model_filter:
            self.logger.info("Filtering for model: %s", self.model_filter)
            # Model filter can match partial model names
            analyzer.df = analyzer.df[
                analyzer.df['agent_model'].str.contains(
                    self.model_filter, case=False, na=False
                )
            ]

        filtered_size = len(analyzer.df)
        self.logger.info(
            "Filtered data: %d -> %d records", original_size, filtered_size
        )

        if filtered_size == 0:
            self.logger.error("No data remaining after filtering!")
            return False

        # Update plots directory to reflect filtering
        if self.game_filter or self.model_filter:
            filter_suffix = []
            if self.game_filter:
                filter_suffix.append(f"game_{self.game_filter}")
            if self.model_filter:
                filter_suffix.append(f"model_{self.model_filter}")

            self.plots_dir = self.plots_dir / "_".join(filter_suffix)
            self.plots_dir.mkdir(exist_ok=True)
            self.logger.info("Plots will be saved to: %s", self.plots_dir)

        return True

    def step_2_reasoning_analysis(self, merged_csv_path: str) -> bool:
        """Step 2: Run comprehensive reasoning analysis."""
        step_name = "Reasoning Analysis"
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 2: %s", step_name)
        self.logger.info("="*60)

        try:
            # Initialize analyzer
            analyzer = LLMReasoningAnalyzer(merged_csv_path)

            # Apply filters if specified
            if not self._apply_filters(analyzer):
                raise ValueError("No data remaining after filtering")

            # Categorize reasoning patterns
            self.logger.info("Categorizing reasoning patterns...")
            analyzer.categorize_reasoning()

            # Save the categorized data back to CSV for future use
            self.logger.info("Saving categorized data back to CSV...")
            analyzer.df.to_csv(merged_csv_path, index=False)

            # Generate metrics and plots
            self.logger.info(
                "Computing metrics and generating visualizations...")
            analyzer.compute_metrics(plot_dir=str(self.plots_dir))

            # Generate additional visualizations (if enabled)
            # HEATMAPS DISABLED
            # try:
            #     self.logger.info("Generating heatmaps...")
            #     analyzer.plot_heatmaps_by_agent(output_dir=str(self.plots_dir))
            # except Exception as e:
            #     self.logger.warning("Heatmap generation failed: %s", e)

            try:
                self.logger.info("Generating entropy trend lines...")
                analyzer.plot_entropy_trendlines()
            except Exception as e:
                self.logger.warning("Entropy trend generation failed: %s", e)
            # Additional entropy plots
            try:
                self.logger.info("Generating entropy by turn across agents...")
                analyzer.plot_entropy_by_turn_across_agents(
                    output_dir=str(self.plots_dir))
            except Exception as e:
                self.logger.warning(
                    "plot_entropy_by_turn_across_agents failed: %s", e)

            try:
                self.logger.info("Generating average entropy across games...")
                analyzer.plot_avg_entropy_across_games(
                    output_dir=str(self.plots_dir))
            except Exception as e:
                self.logger.warning(
                    "plot_avg_entropy_across_games failed: %s", e)

            # Count generated plot files
            plot_files = list(self.plots_dir.glob("*.png"))
            self.pipeline_results["files_generated"].extend(
                [str(f) for f in plot_files])

            self.pipeline_results["summary"]["reasoning_analysis"] = {
                "total_plot_files": len(plot_files),
                "analyzer_initialized": True
            }

            self.log_step(
                step_name, "COMPLETED",
                "Generated %d visualization files in %s" % (
                    len(plot_files), self.plots_dir)
            )

            return True

        except Exception as e:
            self.log_step(step_name, "FAILED", str(e))
            return False

    def step_3_generate_plots(self, merged_csv_path: str) -> bool:
        """Step 3: Generate comprehensive reasoning plots."""
        step_name = "Comprehensive Plot Generation"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STEP 3: {step_name}")
        self.logger.info(f"{'='*60}")

        try:
            # Import the plotting classes directly
            from generate_reasoning_plots import ReasoningPlotGenerator

            self.logger.info(
                "Initializing plot generator with data: %s", merged_csv_path)

            # Initialize the plotter
            plotter = ReasoningPlotGenerator(merged_csv_path)

            # Generate comprehensive plots
            self.logger.info("Generating basic reasoning analysis plots...")
            basic_files = plotter.generate_model_plots(str(self.plots_dir))

            self.logger.info("Generating enhanced evolution plots...")
            evolution_files = plotter.plot_all_reasoning_evolutions_enhanced(
                str(self.plots_dir))

            self.logger.info("Generating standard evolution plots...")
            std_evolution_files = plotter.plot_all_reasoning_evolutions(
                str(self.plots_dir))

            self.logger.info("Analyzing reasoning evolution patterns...")
            evolution_patterns = plotter.analyze_reasoning_evolution_patterns()
            # Save evolution pattern summary
            evolution_summary_path = (
                self.plots_dir / "reasoning_evolution_patterns.json")
            with open(evolution_summary_path, "w", encoding='utf-8') as f:
                json.dump(json_serializable(evolution_patterns), f, indent=2)

            # Optionally, describe evolution pattern for first model/game
            try:
                for model, model_data in evolution_patterns.get(
                        'models', {}).items():
                    for game, game_data in model_data.get('games', {}).items():
                        desc = plotter._describe_evolution_pattern(
                            game_data['dominant_by_turn'])
                        self.logger.info(
                            "Evolution pattern for %s - %s: %s",
                            model, game, desc)
                        break
                    break
            except Exception as e:
                self.logger.warning(
                    "_describe_evolution_pattern failed: %s", e)

            # Count total generated files
            all_generated_files = (
                basic_files + evolution_files + std_evolution_files +
                [str(evolution_summary_path)])

            # Log summary
            self.logger.info(
                "✅ Generated %d plot files:", len(all_generated_files))
            self.logger.info(
                "   • Basic plots (bar, pie, stacked): %d", len(basic_files))
            self.logger.info(
                "   • Evolution plots (enhanced): %d", len(evolution_files))
            self.logger.info(
                "   • Evolution plots (standard): %d",
                len(std_evolution_files))
            self.logger.info(
                "   • Evolution pattern summary: %s", evolution_summary_path)

            # Add to pipeline results
            self.pipeline_results["files_generated"].extend(
                all_generated_files)
            self.pipeline_results["summary"]["comprehensive_plots"] = {
                "total_files": len(all_generated_files),
                "basic_plots": len(basic_files),
                "evolution_plots": len(evolution_files),
                "std_evolution_plots": len(std_evolution_files),
                "output_directory": str(self.plots_dir)
            }

            self.log_step(
                step_name, "COMPLETED",
                f"Generated {len(all_generated_files)} comprehensive plots in "
                f"{self.plots_dir}")

            return True

        except Exception as e:
            self.log_step(step_name, "FAILED", str(e))
            self.logger.error(f"Plot generation failed: {e}")
            return False

    def step_4_extract_sample_traces(self) -> bool:
        """Step 4: Extract and export sample reasoning traces."""
        step_name = "Trace Extraction"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STEP 4: {step_name}")
        self.logger.info(f"{'='*60}")

        try:
            # Find the latest database file (excluding human player data)
            db_files = list(self.results_dir.glob("*.db"))
            # Filter out human player databases
            db_files = [db for db in db_files
                        if not db.stem.startswith("human")]
            if not db_files:
                self.logger.warning(
                    "No database files found for trace extraction")
                return True  # Not critical failure

            latest_db = max(db_files, key=lambda x: x.stat().st_mtime)

            # Create sample exports directory
            exports_dir = self.plots_dir / "sample_exports"
            exports_dir.mkdir(exist_ok=True)

            # Export sample traces to different formats
            sample_csv = exports_dir / "sample_reasoning_traces.csv"
            sample_txt = exports_dir / "detailed_reasoning_report.txt"

            # Actually run extraction and export using
            # extract_reasoning_traces.py functions
            try:
                from extract_reasoning_traces import (
                    extract_reasoning_traces,
                    export_traces_csv,
                    export_traces_txt
                )
                df = extract_reasoning_traces(str(latest_db))
                if df is not None:
                    export_traces_csv(df, str(sample_csv))
                    export_traces_txt(df, str(sample_txt))
                    self.logger.info(
                        "Sample trace files generated in %s", exports_dir)
                    self.pipeline_results["files_generated"].extend(
                        [str(sample_csv), str(sample_txt)])
                    self.log_step(
                        step_name, "COMPLETED",
                        f"Sample traces extracted to {exports_dir}")
                    return True
                else:
                    self.logger.warning("No traces extracted from database.")
                    self.log_step(
                        step_name, "FAILED",
                        "No traces extracted from database."
                    )
                    return False
            except Exception as e:
                self.logger.error("Trace extraction failed: %s", e)
                self.log_step(step_name, "FAILED", str(e))
                return False

        except Exception as e:
            self.log_step(step_name, "FAILED", str(e))
            return False

    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report of the analysis."""
        report_path = self.plots_dir / "analysis_summary_report.json"

        self.pipeline_results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Add pipeline statistics
        total_steps = 4
        completed_steps = len(self.pipeline_results["steps_completed"])
        failed_steps = len(self.pipeline_results["steps_failed"])

        self.pipeline_results["summary"]["pipeline_stats"] = {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "success_rate": f"{(completed_steps/total_steps)*100:.1f}%",
            "total_files_generated": len(
                self.pipeline_results["files_generated"]
            )
        }

        # Save detailed report
        with open(report_path, 'w') as f:
            json.dump(json_serializable(self.pipeline_results), f, indent=2)

        self.logger.info("\n" + "="*60)
        self.logger.info("ANALYSIS PIPELINE SUMMARY")
        self.logger.info("="*60)
        self.logger.info(
            "Pipeline completed: %d/%d steps successful",
            completed_steps, total_steps
        )
        self.logger.info(
            "Files generated: %d",
            len(self.pipeline_results['files_generated'])
        )
        self.logger.info("Output directory: %s", self.plots_dir)
        self.logger.info("Detailed report: %s", report_path)

        if failed_steps > 0:
            self.logger.warning(
                "⚠️  %d steps failed - check logs for details", failed_steps)
        else:
            self.logger.info("✅ All analysis steps completed successfully!")

        return str(report_path)

    def run_full_pipeline(self) -> bool:
        """Run the complete analysis pipeline."""
        self.pipeline_results["start_time"] = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.logger.info("\n🚀 Starting Game Reasoning Arena Analysis Pipeline")
        self.logger.info("Results directory: %s", self.results_dir)
        self.logger.info("Output directory: %s", self.plots_dir)
        self.logger.info("Verbose mode: %s", self.verbose)
        success = True
        # Step 1: Merge databases
        merged_csv = self.step_1_merge_databases()
        if not merged_csv:
            self.logger.error("❌ Pipeline failed at Step 1 (Database Merging)")
            success = False
            merged_csv = None
        # Step 2: Reasoning analysis (only if Step 1 succeeded)
        if merged_csv and not self.step_2_reasoning_analysis(merged_csv):
            self.logger.error(
                "❌ Pipeline failed at Step 2 (Reasoning Analysis)"
            )
            success = False
        # Step 3: Generate additional plots (continue even if Step 2 failed)
        if merged_csv and not self.step_3_generate_plots(merged_csv):
            self.logger.warning("⚠️  Step 3 (Plot Generation) had issues")
        # Step 4: Extract sample traces (continue regardless)
        if not self.step_4_extract_sample_traces():
            self.logger.warning("⚠️  Step 4 (Trace Extraction) had issues")
        # Generate summary report
        self.generate_summary_report()
        return success


def main():
    """Main entry point for the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete Game Reasoning Arena analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full analysis with default settings
    python analysis/run_full_analysis.py

    # Specify custom directories
    python analysis/run_full_analysis.py --results-dir results \
        --plots-dir custom_plots

    # Run analysis only for HEX game
    python analysis/run_full_analysis.py --game hex

    # Run analysis only for a specific model
    python analysis/run_full_analysis.py --model llama3

    # Run analysis for HEX game with specific model
    python analysis/run_full_analysis.py --game hex --model llama3

    # Run in quiet mode
    python analysis/run_full_analysis.py --quiet

    # Skip existing files
    python analysis/run_full_analysis.py --skip-existing
        """
    )

    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing SQLite database files (default: results)"
    )

    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory for output plots and visualizations (default: plots)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run in quiet mode (less verbose output)"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip analysis steps if output files already exist"
    )

    parser.add_argument(
        "--game",
        help="Filter analysis for a specific game (e.g., hex, tic_tac_toe)"
    )

    parser.add_argument(
        "--model",
        help="Filter analysis for a specific model (e.g., llama3, gpt-4)"
    )

    args = parser.parse_args()

    # Initialize and run pipeline
    pipeline = AnalysisPipeline(
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        verbose=not args.quiet,
        skip_existing=args.skip_existing,
        game_filter=args.game,
        model_filter=args.model
    )

    success = pipeline.run_full_pipeline()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
