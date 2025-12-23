"""
Manual test script for fetching grid weather data.

This script demonstrates how to use the `fetch_grid_weather` function to download weather data for a 4x4 grid of latitude and longitude points from the NASA POWER API.

**Note:** This script will make external API calls to the NASA POWER API and will download approximately 16 CSV files, each containing weather data from 2005 to 2025. This may take some time and consume a significant amount of bandwidth.
"""

import asyncio
from pathlib import Path
import sys
import os

# Add repo to path
repo_path = Path(__file__).parent.parent
sys.path.insert(0, str(repo_path))

from app.data_fetcher import fetch_grid_weather

def test_fetch_grid_weather():
    """
    Demonstrates how to use the `fetch_grid_weather` function.
    """
    print("=" * 70)
    print("Testing Grid Weather Data Fetching")
    print("=" * 70)

    output_dir = os.path.join(repo_path, 'data', 'grid-weather-test')

    print(f"This test will fetch weather data and save it to '{output_dir}'.")
    print("This will download 16 CSV files and may take a few minutes.")
    print("-" * 70)
    
    # The following line is commented out to prevent the test from making actual API calls.
    # You can uncomment it to run the data fetching process.
    
    # print("To run the test, uncomment the 'asyncio.run' line in the 'if __name__ == ""__main__""' block.")

if __name__ == "__main__":
    # To run this test, uncomment the following line.
    # Note: This will make external API calls and download data.
    asyncio.run(fetch_grid_weather(output_dir=os.path.join(Path(__file__).parent.parent, 'data', 'grid-weather-test')))
    
    print("This script is a demonstration. To fetch the data, please uncomment the asyncio.run line in the script.")
