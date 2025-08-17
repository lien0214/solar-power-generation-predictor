import requests
import pandas as pd
import io
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import fetch_pm_name_id

class WeatherClien:
    params: list[str]

    DAILY_POINT_URL: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    MAX_PARAMS: int = 20
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0
    PROBLEMATIC_PARAMS: set = {"PSC", "WSC"}  # Parameters that require additional elevation/surface data

    def __init__(self, community: str, temporal: str):
        all_params = fetch_pm_name_id.get_nasa_parameters("SB", "daily")
        self.params = [p for p in all_params if p not in self.PROBLEMATIC_PARAMS]

    def fetch_all_columns(self, long: float, lat: float, dateFrom: str, dateTo: str) -> pd.DataFrame:
        return self._fetch_columns_by_batch(long, lat, dateFrom, dateTo, self.params)

    def fetch_columns(self, long: float, lat: float, dateFrom: str, dateTo: str, columns: list[str]) -> pd.DataFrame:
        columns = [col for col in columns if col in self.params]
        if not columns:
            raise ValueError("No valid columns provided.")
        return self._fetch_columns_by_batch(long, lat, dateFrom, dateTo, columns)

    def _fetch_columns_by_batch(self, long: float, lat: float, dateFrom: str, dateTo: str, columns: list[str]) -> pd.DataFrame:
        batches = []
        total_batches = (len(columns) + self.MAX_PARAMS - 1) // self.MAX_PARAMS
        
        for i in range(0, len(columns), self.MAX_PARAMS):
            batch_params = columns[i:i + self.MAX_PARAMS]
            batch_num = i // self.MAX_PARAMS + 1
            batches.append((batch_num, batch_params))
        
        def fetch_batch_with_retry(batch_info):
            batch_num, batch_params = batch_info
            
            for attempt in range(self.MAX_RETRIES):
                try:
                    print(f"\rFetching batch {batch_num}/{total_batches} ({len(batch_params)} parameters)...", end="", flush=True)
                    df_batch = self._fetch(long, lat, dateFrom, dateTo, batch_params)
                    print(f"\rBatch {batch_num}/{total_batches} completed successfully                    ")
                    return df_batch
                    
                except Exception as e:
                    if attempt < self.MAX_RETRIES - 1:
                        print(f"\rBatch {batch_num}/{total_batches} failed, retrying...                    ", end="", flush=True)
                        time.sleep(self.RETRY_DELAY)
                    else:
                        print(f"\rBatch {batch_num}/{total_batches} failed after {self.MAX_RETRIES} attempts")
                        raise
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_batch = {executor.submit(fetch_batch_with_retry, batch): batch for batch in batches}
            all_dfs = []
            
            for future in as_completed(future_to_batch):
                try:
                    df_batch = future.result()
                    all_dfs.append(df_batch)
                except Exception as e:
                    batch_num = future_to_batch[future][0]
                    print(f"Batch {batch_num} failed permanently")
        
        if not all_dfs:
            raise ValueError("No data could be fetched from any batch")
        
        result_df = all_dfs[0]
        for df in all_dfs[1:]:
            result_df = result_df.join(df, how='outer')
        
        return result_df

    def _fetch(self, long: float, lat: float, dateFrom: str, dateTo: str, columns: list[str]) -> pd.DataFrame:
        query = {
            "start": dateFrom,
            "end": dateTo,
            "latitude": lat,
            "longitude": long,
            "community": "SB",
            "parameters": ",".join(columns),
            "format": "csv",
            "header": "false",
        }
        resp = requests.get(self.DAILY_POINT_URL, params=query)
        resp.raise_for_status()
        first_line = next((ln for ln in resp.text.splitlines() if ln.strip()), "")
        cols = [c.strip() for c in first_line.split(",")]

        df = pd.read_csv(io.StringIO(resp.text), names=cols)

        date_cols = ["YEAR", "MO", "DY"]
        if all(col in df.columns for col in date_cols):
            for col in date_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df["date"] = pd.to_datetime(df[["YEAR", "MO", "DY"]].rename(columns={"MO": "month", "DY": "day"}))
            df = df.drop(columns=date_cols).set_index("date").sort_index()

        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

if __name__ == "__main__":
    print("Initializing WeatherAcl...")
    acl = WeatherClien("SB", "daily")
    print(f"Found {len(acl.params)} available parameters (excluded PSC, WSC)")
    
    print("Fetching weather data...")
    df = acl.fetch_all_columns(119.561319, 23.566406, "20250201", "20250430")
    
    output_file = "weather_data.csv"
    df.to_csv(output_file)
    print(f"Data saved to {output_file}")
    print(f"Dataset: {df.shape[0]} days × {df.shape[1]} parameters")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print("Weather data fetch completed successfully!")