# app_config.py
# ⚠️ DO NOT COMMIT THIS FILE TO GITHUB ⚠️

# API Secrets
API_KEY = "your-actual-key-here"
JWT_SECRET = "super-secret-jwt-string"
CACHE_FILE = ""
CACHE_JSON_LOCAL_MODEL = ""
CACHE_JSON_GPT_LARGE = ""
CACHE_JSON_GPT_SMALL = ""
NAME_MODEL_EMBEDDING = ""

from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ADMIM_USER_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("KEY")
    }
}

