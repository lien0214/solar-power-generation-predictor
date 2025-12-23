"""
Manual test script for checking the last valid weather date.

This script reads the historical weather data file and determines the last date with valid data.
It checks for the presence of -999 as an indicator of missing data, as requested by the user.
"""

import pandas as pd
from pathlib import Path
import sys

# Define missing value constant consistent with the project
MISSING_VALUE = -999

def test_check_last_weather_date():
    """
    Checks and prints the last valid weather date from the historical weather file.
    """
    print("=" * 70)
    print("Testing Check Last Weather Date")
    print("=" * 70)

    # Define the path to the weather file relative to this script
    script_dir = Path(__file__).parent
    weather_file = script_dir.parent / "code" / "data" / "23.530236_119.588339.csv"
    print(f"Reading historical weather data from: {weather_file}")

    try:
        df = pd.read_csv(weather_file)
        df['Date'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}))

        # Check for -999 values
        # Exclude 'Date' column from the check to avoid comparing datetime objects with int
        check_df = df.drop(columns=['Date'])
        if (check_df == MISSING_VALUE).any().any():
            print(f"Found {MISSING_VALUE} values in the data.")
            # Find the last row that does not contain -999
            last_valid_index = df.index[(check_df != MISSING_VALUE).all(axis=1)].max()
            if pd.notna(last_valid_index):
                last_valid_date = df.loc[last_valid_index, 'Date']
                print(f"✅ Last valid weather date (before {MISSING_VALUE}): {last_valid_date.date()}")
            else:
                print(f"⚠️ Could not determine a valid date before the {MISSING_VALUE} values.")
        else:
            print(f"No {MISSING_VALUE} values found in the data.")
            last_valid_date = df['Date'].max()
            print(f"✅ Last valid weather date: {last_valid_date.date()}")

    except FileNotFoundError:
        print(f"❌ Error: Weather data file not found at '{weather_file}'")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    test_check_last_weather_date()
