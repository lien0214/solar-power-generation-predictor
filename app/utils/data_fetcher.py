"""
Data fetching utilities.
"""
import os
import csv
import asyncio
import aiohttp
import async_timeout
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF = 2
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = (
    "T2M,T2M_MAX,TS,CLOUD_AMT_DAY,CLOUD_OD,ALLSKY_SFC_SW_DWN,RH2M,ALLSKY_SFC_SW_DIRH"
)
COMMUNITY = "sb"
FORMAT = "csv"
HEADER = "false"
CONCURRENT_REQUESTS = 8

def generate_grid(lat_min: float, lat_max: float, lon_min: float, lon_max: float, size: int) -> List[Tuple[float, float]]:
    """
    Generate a grid of latitude and longitude points.
    """
    lat_step = (lat_max - lat_min) / (size - 1)
    lon_step = (lon_max - lon_min) / (size - 1)
    lat_points = [round(lat_min + i * lat_step, 6) for i in range(size)]
    lon_points = [round(lon_min + i * lon_step, 6) for i in range(size)]
    return [(lat, lon) for lat in lat_points for lon in lon_points]


def add_lat_lon_columns(content: str, lat: float, lon: float) -> List[List[str]]:
    """
    Add LAT and LON columns to the CSV content.
    """
    lines = content.strip().splitlines()
    reader = csv.reader(lines)
    rows = list(reader)

    header = rows[0] + ["LAT", "LON"]
    updated_rows = [row + [str(lat), str(lon)] for row in rows[1:]]

    return [header] + updated_rows


async def _fetch_and_save(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, lat: float, lon: float, start_date: str, end_date: str, output_dir: str):
    """
    Fetch data for a single point and save it to a CSV file.
    """
    params = {
        "start": start_date,
        "end": end_date,
        "latitude": lat,
        "longitude": lon,
        "community": COMMUNITY,
        "parameters": PARAMETERS,
        "format": FORMAT,
        "header": HEADER,
    }

    url = BASE_URL
    filename = f"{lat}_{lon}.csv"
    filepath = os.path.join(output_dir, filename)

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with async_timeout.timeout(30):
                    async with session.get(url, params=params) as response:
                        response.raise_for_status()
                        text = await response.text()
                        rows = add_lat_lon_columns(text, lat, lon)

                        with open(filepath, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerows(rows)

                        logger.info(f"Saved: {filename}")
                        return

            except Exception as e:
                logger.error(f"[Attempt {attempt}] Error for {lat}, {lon}: {e}")
                if attempt < MAX_RETRIES:
                    backoff = INITIAL_BACKOFF ** attempt
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"Failed after {MAX_RETRIES} attempts for lat={lat}, lon={lon}")


async def fetch_grid_weather(
    output_dir: str,
    lat_min: float = 23.199836,
    lat_max: float = 23.758598,
    lon_min: float = 119.312190,
    lon_max: float = 119.692245,
    grid_size: int = 4,
    start_date: str = "20050101",
    end_date: str = "20251105",
):
    """
    Fetch weather data for a grid of latitude and longitude points.
    """
    os.makedirs(output_dir, exist_ok=True)
    grid = generate_grid(lat_min, lat_max, lon_min, lon_max, grid_size)
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:
        tasks = [
            _fetch_and_save(session, semaphore, lat, lon, start_date, end_date, output_dir)
            for lat, lon in grid
        ]
        await asyncio.gather(*tasks)
