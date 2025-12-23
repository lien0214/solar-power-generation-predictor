# 太陽能發電量預測器

歡迎來到太陽能發電量預測器專案！此儲存庫包含一個可用於生產環境的 FastAPI 應用程式，旨在使用機器學習預測太陽能發電量。

## 🚀 專案概覽

本系統可預測特定地點和日期的太陽能發電量（單位為 kWh）。它採用混合式數據管線，結合歷史天氣數據與滾動式機器學習預測，為太陽能預測模型提供準確的輸入。

**主要功能：**
*   **混合式天氣管線 (Hybrid Weather Pipeline)**：在歷史數據（模擬 API 抓取）和未來的滾動預測之間無縫切換。
*   **雙重預測策略 (Dual Prediction Strategies)**：
    *   `merged`: A generalist model trained on all sites.
    *   `seperated`: An ensemble of site-specific models (averaged).
*   **動態訓練 (Dynamic Training)**：自動發現並訓練放置於資料夾中的新太陽能案場數據。
*   **FastAPI & XGBoost**：提供梯度提升模型的高效能 API。

## 📚 文件

詳細文件位於 `doc/` 目錄中：

*   **安裝指南 (Setup Guide)**：安裝、配置及運行應用程式。
*   **API 合約 (API Contract)**：詳細的端點規格（日、月、年）。
*   **系統架構 (System Architecture)**：高層次設計、數據流與組件圖。
*   **測試指南 (Testing Guide)**：如何運行完整的測試套件。
*   **問題排解 (Troubleshooting)**：常見問題与解決方案。
*   **技術棧 (Tech Stack)**：使用的函式庫與工具。

### 組件深入探討
*   模型管理器 (Model Manager)
*   預測引擎 (Prediction Engine)
*   天氣抓取器 (Weather Fetcher)

## 🛠️ 快速入門

### 1. 先決條件
*   Python 3.9+
*   `pip`

### 2. 安裝
```bash
cd repo
pip install -r requirements.txt
```

### 3. Running the App
You can run the application in two modes (controlled by `STARTUP_MODE`):

**Option A: Load Pre-trained Models (Fast)**
```bash
# Default mode - loads models from ./models/
uvicorn main:app --reload
```

**Option B: Train from Scratch**
```bash
# Retrains models using data in app/data/
STARTUP_MODE=train_now uvicorn main:app --reload
```

### 4. Explore the API
Once running, open your browser to:
*   **Swagger UI**: http://127.0.0.1:8000/docs
*   **ReDoc**: http://127.0.0.1:8000/redoc

## 📂 Repository Structure

```
repo/
├── app/data/           # Solar training data (CSVs)
├── doc/                # Documentation
├── manual_testing/     # Scripts for manual verification
├── models/             # Saved ML models (.pkl)
├── main.py             # Application entry point
├── requirements.txt    # Dependencies
└── README.md           # This file
```

For more details on testing, run `pytest` or check the Testing Guide.
