Here is the full content you can copy and paste into your `README.md` file:

```markdown
# 🚀 Product Catalog API

A modern, asynchronous **Product Catalog API** built with **FastAPI**, **PostgreSQL**, and **JWT authentication**.  
Includes secure user authentication, admin-protected endpoints, and complete Swagger API documentation.

---

## 📁 Project Structure
```

project/
│
├── app/
│ ├── auth.py # Auth logic (JWT, hash, current user)
│ ├── database.py # Database connection
│ ├── main.py # App entrypoint
│ ├── models.py # SQLAlchemy models
│ ├── schemas.py # Pydantic models
│ └── routes/
│ ├── users.py
│ ├── categories.py
│ └── products.py
│
├── requirements.txt
├── .env
└── README.md

````

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/product-catalog-api.git
cd product-catalog-api
````

### 2. Setup Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # On Linux/macOS
.venv\\Scripts\\activate       # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```
SECRET_KEY=your_secret_key_here
DATABASE_URL=postgresql+asyncpg://postgres:yourpassword@localhost/e-commerce
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

---

## 🏃 Run the App

```bash
uvicorn app.main:app --reload
```

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Redoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔐 Authentication

- Register: `POST /auth/register`
- Login: `POST /auth/login` → Get JWT token
- Use token as `Authorization: Bearer <your_token>` for secured endpoints

---

## 🛠 Admin-Only Endpoints

The following routes are only accessible to admin users (`is_admin = True`):

- Categories: `POST /categories/`, `PUT`, `DELETE`
- Products: `POST /products/`, `PUT`, `DELETE`

---

## 🧪 Testing

You can test endpoints using:

- [Swagger UI](http://localhost:8000/docs)
- [Hoppscotch](https://hoppscotch.io/)
- [Postman](https://www.postman.com/)
