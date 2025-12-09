## 💡 Problem & Tech Stack

| Category | Problem / Requirement | Recommended Solution |
| :--- | :--- | :--- |
| **Core Function** | Fetch remote data & train ML model (or load) *once* on startup, then serve predictions for a year. | **Startup Event** in FastAPI (`@app.on_event("startup")` or `lifespan`) to handle one-time I/O and model loading. |
| **Backend API** | Light, feature-first, develop-once, low-scale server. | **FastAPI** (Python) for high performance and rapid development with type-checking (Pydantic). |
| **Frontend/Client** | Easy, packed client interface (like Swagger) for client interaction. | **Automatic Swagger UI** (OpenAPI) built into FastAPI (`/docs` endpoint). |
| **Distribution** | Single, double-clickable executable (Windows/Mac/Linux). | **PyInstaller** to package the FastAPI app, Uvicorn server, and Python environment into a standalone file. |

**Concise Stack:** **FastAPI (Python)** for the API, **Uvicorn** as the server, and **PyInstaller** for the executable.
