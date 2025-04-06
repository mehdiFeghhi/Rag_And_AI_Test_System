from app_config import ADMIM_USER_DB,pwd_context

admin_users_db = ADMIM_USER_DB

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user_by_username(username: str):
    return admin_users_db.get(username)

def authenticate_admin(username: str, password: str):
    user = get_user_by_username(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user
