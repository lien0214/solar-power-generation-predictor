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

### 2. 建立和激活虛擬環境

建議為專案建立一個虛擬環境，以管理依賴項並避免與其他專案發生衝突。

```bash
# 建立虛擬環境
python3 -m venv .venv

# 激活虛擬環境 (macOS/Linux)
source .venv/bin/activate

# 激活虛擬環境 (Windows PowerShell)
# .\.venv\Scripts\Activate.ps1

# 激活虛擬環境 (Windows Command Prompt)
# .venv\Scripts\activate.bat
```

### 3. 安裝依賴項
```bash
cd repo
pip install -r requirements.txt
```

### 4. 環境變數設定

此專案使用環境變數進行配置。您可以從 `.env.example` 檔案中複製範本並進行修改。

```bash
cp .env.example .env
```

開啟 `.env` 檔案並根據您的需求調整變數。特別注意 `STARTUP_MODE` 變數，它控制應用程式啟動時是訓練模型 (`train_now`) 還是加載預訓練模型 (`load_models`)。

### 5. 運行應用程式
您可以透過兩種模式運行應用程式（由 `STARTUP_MODE` 控制）：

#### 選項 A: 加載預訓練模型 (快速)
```bash
# 預設模式 - 從 ./models/ 加載模型
uvicorn main:app --reload
```

#### 選項 B: 從頭開始訓練
```bash
# 使用 app/data/ 中的數據重新訓練模型
STARTUP_MODE=train_now uvicorn main:app --reload
```

### 6. 探索 API
應用程式運行後，請在瀏覽器中打開：
*   **Swagger UI**: http://127.0.0.1:8000/docs
*   **ReDoc**: http://127.0.0.1:8000/redoc

## 📂 儲存庫結構

```
repo/
├── app/data/           # 太陽能訓練數據 (CSVs)
├── doc/                # 文件
├── manual_testing/     # 手動驗證腳本
├── models/             # 已保存的機器學習模型 (.pkl)
├── main.py             # 應用程式入口點
├── requirements.txt    # 依賴項
└── README.md           # 此檔案
```

有關測試的更多詳細資訊，請運行 `pytest` 或查閱 測試指南。
