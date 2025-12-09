# Cache Layer Component

## Purpose
Stores and retrieves prediction results and intermediate data using Redis (local or remote, configurable).

## Key Functions
- `get(key: str) -> Any`
  - Retrieves cached value for a key.
- `set(key: str, value: Any, expire: int = None) -> None`
  - Stores value in cache with optional expiration.
- `delete(key: str) -> None`
  - Removes a key from cache.

## Example Usage
```python
cache.set('prediction:day:2025-01-01', result, expire=3600)
cached = cache.get('prediction:day:2025-01-01')
```

## Inputs/Outputs
- **Inputs**: Key, value, expiration
- **Outputs**: Cached data for fast access
