"""
agent_report.py

Generates reports and visualizations for agent performance.
"""

#TODO: this seems to be not used!

import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from .base_agent import BaseAgent

class AgentPerformanceReporter:
    """
    Collects and visualizes performance metrics for multiple agents.
    """

    def __init__(self, agents: List[BaseAgent], report_file="agent_performance.csv"):
        """
        Args:
            agents (List[BaseAgent]): A list of agents to track.
            report_file (str): The name of the CSV file to store the results.
        """
        self.agents = agents
        self.report_file = report_file

    def collect_metrics(self):
        """
        Collects performance data from all agents and saves to a CSV file.
        """
        data = []

        for agent in self.agents:
            metrics = agent.get_performance_metrics()
            data.append(metrics)

        df = pd.DataFrame(data)
        df.to_csv(self.report_file, index=False)
        print(f"Performance data saved to {self.report_file}")

    def print_summary(self):
        """
        Prints a summary of agent performance.
        """
        print("\n--- Agent Performance Summary ---")
        for agent in self.agents:
            metrics = agent.get_performance_metrics()
            print(f"{agent.agent_type}: {metrics}")

    def plot_metrics(self):
        """
        Generates visualizations comparing agent performance.
        """
        df = pd.read_csv(self.report_file)

        # Bar Chart: Average Response Time per Agent
        plt.figure(figsize=(8, 5))
        plt.bar(df["agent_type"], df["average_response_time"], alpha=0.7)
        plt.xlabel("Agent Type")
        plt.ylabel("Average Response Time (s)")
        plt.title("Agent Average Response Time Comparison")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.show()

        # Histogram: Total Actions Taken per Agent
        plt.figure(figsize=(8, 5))
        plt.bar(df["agent_type"], df["actions_taken"], alpha=0.7, color="green")
        plt.xlabel("Agent Type")
        plt.ylabel("Total Actions Taken")
        plt.title("Total Actions Taken by Each Agent")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.show()
