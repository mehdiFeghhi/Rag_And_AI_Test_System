from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from starlette.status import HTTP_302_FOUND
from pydantic import BaseModel

from security.admin import authenticate_admin
from security.auth import create_access_token, get_current_user


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---------------------------
# Login and Dashboard Routes
# ---------------------------


@app.get("/", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/token")
async def login_for_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_admin(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_access_token(data={"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: dict = Depends(get_current_user)):
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user["username"]})


# ---------------------------
# Chat API Endpoint
# ---------------------------

class ChatRequest(BaseModel):
    query: str
    top_k: int = 3  # Default number of retrieval results

@app.post("/chat")
async def chat_endpoint(chat: ChatRequest, user: dict = Depends(get_current_user)):
    """
    AI Chat endpoint that accepts a query and a top_k parameter.
    Returns:
      - "retrieval": a list of fake retrieval results, repeated top_k times.
      - "main": a fake main AI response.
    """
    # Create a list of fake retrieval results based on top_k.
    retrieval_results = [
        f"Fake retrieval result {i+1} for query: '{chat.query}'"
        for i in range(chat.top_k)
    ]
    
    main_result = f"Fake main AI response for query: '{chat.query}'"
    
    return {"retrieval": retrieval_results, "main": main_result}
