## Agent Rules for Professional FastAPI Structure

This document defines the strict responsibilities (Agent Rules) for each component package within the application structure. This structure enforces **Separation of Concerns (SoC)**, ensuring maintainability, testability, and scalability.

---

### I. Routing and API Layer (The Gateway)

| Package | Agent Rule (Responsibility) | Key Artifacts |
| :--- | :--- | :--- |
| **`app/api/v1/`** | **HTTP Communication Handler.** This agent receives the incoming HTTP request, validates the input using Pydantic schemas defined in `app/schemas/`, calls the appropriate **Service Agent** (from `app/services/`), and formats/returns the final HTTP response (JSON, status codes). **It must NOT contain business logic or direct database queries.** | `endpoints.py`, `APIRouter` objects. |
| **`app/main.py`** | **The Application Orchestrator.** This agent creates the primary `FastAPI` application instance, configures global middleware, registers event handlers, and includes all the `APIRouter` objects aggregated from the `api/` layer. | `FastAPI()` instance. |

### II. Business Logic Layer (The Brain)

| Package | Agent Rule (Responsibility) | Key Artifacts |
| :--- | :--- | :--- |
| **`app/services/`** | **The Business Expert.** This agent contains all complex **business logic, transactional workflows, and domain-specific validation rules.** It acts as the mediator, taking data from the API agent and coordinating operations with the data access agent (`crud/`). **It must NOT be aware of HTTP requests, responses, or routing.** | `user_service.py`, `item_service.py` (Classes/functions containing complex logic). |

### III. Data Access and Models Layer (The Memory)

| Package | Agent Rule (Responsibility) | Key Artifacts |
| :--- | :--- | :--- |
| **`app/crud/`** | **The Data Handler/Repository.** This agent is solely responsible for direct, atomic database interaction (Create, Read, Update, Delete). It operates on the database models (`app/models/`) and is called **only** by the **Service Agent**. It abstracts the ORM details from the business logic. | `base.py`, `user.py` (CRUD functions). |
| **`app/models/`** | **The Table Blueprint.** This agent defines the persistent structure of the application's data as database tables (Entities/Models), typically using an ORM like SQLAlchemy or Tortoise-ORM. | `user.py`, `item.py` (ORM definitions). |
| **`app/schemas/`** | **The Data Format Negotiator (DTOs).** This agent defines the Pydantic models used to enforce the data contract. This includes validating data received in requests (`*Create`, `*Update`) and serializing data sent in responses (`*InDB`, `*Response`). | `user.py` (`UserCreate`, `UserInDB`). |

### IV. Configuration and Utilities (The Tools)

| Package | Agent Rule (Responsibility) | Key Artifacts |
| :--- | :--- | :--- |
| **`app/core/`** | **The Settings Manager.** This agent handles the retrieval and typing of application-wide configuration variables, primarily through environment files (`.env`) using Pydantic's `Settings` class. | `config.py` (Pydantic settings model). |
| **`app/database/`** | **The Connection Manager.** This agent handles the initiation and management of database connections, ORM sessions, and transaction scopes. | `session.py` (DB engine/session management). |
| **`app/utils/`** | **The Shared Toolkit.** This agent houses reusable helper functions, security implementations (password hashing, JWT token generation), and shared constants that are required across multiple primary agents. | `security.py`, `constants.py`, various helper functions. |

---

Would you like to proceed with generating the content for a key file, such as the `app/core/config.py` file, based on the environment variables defined in the last step?