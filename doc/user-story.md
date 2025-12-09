# User Story

## Overview
This product is designed for solar power project buyers/operators who need reliable, on-demand solar generation predictions for specific locations and time ranges. The user interacts with a backend engine via API or Swagger UI, starting the engine, configuring settings, and requesting predictions as needed.

## Actors
- **Project Buyer/Operator**: Starts the engine, configures settings, and requests predictions.
- **System**: Handles data fetching, model training/loading, prediction, caching, and serving API requests.

## Typical Workflow
1. **Configuration**: User sets up product constants and chooses startup mode (train or load models).
2. **Startup**: User starts the engine. The system fetches grid weather data and trains or loads models.
3. **Prediction Requests**: User sends API requests (day, month, year) with location and time parameters.
4. **Result Delivery**: System returns raw predictions, caching results for future requests.

## Value Delivered
- Fast, accurate solar generation predictions for any supported location and time range.
- Simple API interface with Swagger UI for easy client integration.
- Configurable, single-machine deployment with optional Redis caching for performance.
