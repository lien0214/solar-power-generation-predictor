# 太陽能發電量預測系統 (Solar Power Forecasting System)

歡迎使用太陽能發電量預測器！本專案是一個具備生產級標準的 **FastAPI** 應用程式，結合 **XGBoost** 機器學習模型，旨在提供精準的太陽能發電量預測服務。

## 🚀 專案概覽

本系統能根據特定地點與日期，預測太陽能板的發電輸出（單位：kWh）。系統核心採用 **混合式數據管線 (Hybrid Data Pipeline)**，整合歷史氣象觀測值與未來預報數據，確保預測模型在不同時段均能維持高準確度。

**主要功能：**

* **混合式天氣管線 (Hybrid Weather Pipeline)**：自動辨別請求日期，在歷史觀測數據（模擬 API 抓取）與滾動式天氣預報之間無縫切換。
* **雙重預測策略 (Dual Prediction Strategies)**：
* `merged` (通用模型)：使用所有案場數據訓練出的單一模型，具備較佳的泛化能力。
* `separated` (獨立案場模型)：針對個別案場獨立訓練的集成模型（取平均值），提供更高的局部精確度。


* **動態訓練機制 (Dynamic Training)**：系統會自動掃描資料夾，發現新的 CSV 數據後自動執行模型訓練。
* **高效能 API**：基於 FastAPI 與 XGBoost，提供低延遲、高吞吐量的預測接口。

---

## 🛠️ 快速入門

### 1. 先決條件

* Python 3.9 或更高版本
* `pip` 套件管理工具

### 2. 安裝依賴項

`cd repo`
`pip install -r requirements.txt`

### 3. 環境變數設定 & 資料輸入

此專案使用環境變數進行配置。請從 `.env.example` 複製模板並修改。
`cp .env.example .env`

請編輯 `.env` 檔案，特別注意 `STARTUP_MODE` 變數：

* `train_now`: 啟動時立即重新訓練模型。
* `load_models`: 啟動時直接加載預訓練模型。

資料輸入請放在 `./app/data/solar-data/` 資料夾中，並且依照格式取代 `./app/data/solar-data/example.csv`。  
可以放超過一組資料。

#### 海域 ＆ 陸域

若 `./app/data/solar-data/` 中放入海域資料，則模型可以預測海上場域的發電量，反之亦然。若其中一方的資料不及一年，可以放入混合的資料補至超過一年。

### 4. 運行應用程式

您可以透過以下兩種模式運行（取決於 `STARTUP_MODE` 設定）：

**選項 A：快速啟動**
`STARTUP_MODE=load_models uvicorn app.main:app --reload`

**選項 B：重新訓練啟動**
`STARTUP_MODE=train_now uvicorn app.main:app --reload`

### 5. 探索 API 文件

應用程式運行後，您可以訪問以下網址進行測試：

* **Swagger UI (互動式測試)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📂 儲存庫結構說明

**repo/**

* **./app/data/**: 太陽能歷史發電數據 (CSV 檔案)
* **./app/models/**: 已保存的機器學習模型檔案 (.pkl)
* **./app/main.py**: FastAPI 進入點
* **requirements.txt**: Python 套件清單
* **README.md**: 您正在閱讀的文件