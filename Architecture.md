# Scalable FastAPI Architecture (Hybrid src Layout)

This structure blends the standard `src/` layout for packaging with a versioned API router for clean separation of concerns.

## Directory Overview

```text
.
├── src/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # App entrypoint (FastAPI app)
│   │   ├── api/                      # API layer (routers)
│   │   │   ├── __init__.py
│   │   │   ├── deps.py               # Dependencies (e.g., DB session, auth)
│   │   │   ├── api.py                # Central router (includes endpoints)
│   │   │   └── endpoints/
│   │   │       ├── __init__.py
│   │   │       ├── feature_a.py      # e.g., users
│   │   │       └── feature_b.py      # e.g., items
│   │   ├── core/                     # Config & security
│   │   │   ├── __init__.py
│   │   │   ├── config.py             # Pydantic settings / env
│   │   │   └── security.py           # JWT / hashing
│   │   ├── services/                 # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── feature_a_service.py
│   │   │   └── feature_b_service.py
│   │   ├── schemas/                  # Pydantic models (req/resp)
│   │   │   ├── __init__.py
│   │   │   ├── feature_a_schema.py
│   │   │   └── feature_b_schema.py
│   │   ├── models/                   # DB models (SQLAlchemy)
│   │   │   ├── __init__.py
│   │   │   ├── feature_a_model.py
│   │   │   └── feature_b_model.py
│   │   └── db/                       # DB connection
│   │       ├── __init__.py
│   │       └── session.py            # Engine & SessionLocal
├── tests/
│   ├── __init__.py
│   ├── unit/
│   └── integration/
├── .env
├── alembic.ini
└── pyproject.toml
```

## Request–Response Flow

1. Routing (entrypoint)

   - Client calls e.g., `POST /api/items/`.
   - File: `src/app/main.py` initializes the app and includes routers.
   - Flow: `main.py` → `src/app/api/api.py` → endpoint in `src/app/api/endpoints/feature_a.py`.

2. Input validation (schemas)

   - FastAPI validates JSON via Pydantic.
   - File: `src/app/schemas/feature_a_schema.py`.
   - Flow: Endpoint expects `item_in: ItemCreate`. Invalid input returns 422 before endpoint logic runs.

3. Dependencies (DB session)

   - File: `src/app/api/deps.py` and `src/app/db/session.py`.
   - Flow: `db: Session = Depends(deps.get_db)` opens a scoped session per request.

4. Controller (endpoint)

   - File: `src/app/api/endpoints/feature_a.py`.
   - Flow: Endpoint delegates to the service layer, not raw SQL.
   - Example:
     ```python
     # inside endpoint
     return feature_a_service.create_item(db=db, item_in=item_in)
     ```

5. Service layer (business logic)

   - File: `src/app/services/feature_a_service.py`.
   - Flow: Convert request schema → DB model, persist, return domain object.
   - Example:
     ```python
     # inside service
     db_item = ItemModel(**item_in.dict())
     db.add(db_item)
     db.commit()
     return db_item
     ```

6. Persistence (models)

   - File: `src/app/models/feature_a_model.py`.
   - Flow: SQLAlchemy translates Python objects into SQL (INSERT/UPDATE/etc.).

7. Response (serialization)
   - File: `src/app/schemas/feature_a_schema.py`.
   - Flow:
     - Service returns a DB model to the endpoint.
     - Endpoint returns it; FastAPI applies `response_model=ItemResponse`.
     - Sensitive fields are excluded; final JSON is returned.

## Naming Conventions

- Files: snake_case (e.g., `user_profile.py`)
- Classes: PascalCase (e.g., `UserProfile`)
- Variables/Functions: snake_case (e.g., `get_user_profile`)
- Constants: SCREAMING_SNAKE_CASE (e.g., `MAX_RETRY_COUNT`)
