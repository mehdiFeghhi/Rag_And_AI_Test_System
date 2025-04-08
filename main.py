from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from starlette.status import HTTP_302_FOUND
from pydantic import BaseModel
from datetime import datetime, timezone
# from fastapi.staticfiles import StaticFiles  # Import StaticFiles

from bson import ObjectId
from db import chat_collection
from security.admin import authenticate_admin
from security.auth import create_access_token, get_current_user


app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="templates"), name="static")
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


@app.get("/chats_page", response_class=HTMLResponse)
async def chats_page(request: Request, user: dict = Depends(get_current_user)):
    return templates.TemplateResponse("chats.html", {"request": request})


@app.get("/chat_details_page/{chat_id}", response_class=HTMLResponse)
async def chat_details_page(request: Request, chat_id: str, user: dict = Depends(get_current_user)):
    return templates.TemplateResponse("chat_details.html", {"request": request, "chat_id": chat_id})

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

        # Store the chat in MongoDB
    chat_doc = {
        "username": user["username"],
        "query": chat.query,
        "top_k": chat.top_k,
        "retrieval_results": retrieval_results,
        "main_result": main_result,
        "timestamp": datetime.now(timezone.utc)  # Timezone-aware UTC datetime
    }
    await chat_collection.insert_one(chat_doc)

    return {"retrieval": retrieval_results, "main": main_result}


# GET endpoint to retrieve all chats for the current user
@app.get("/chats")
async def get_all_chats(user: dict = Depends(get_current_user)):
    """
    Retrieve all chat documents associated with the current user.
    """
    chats = []
    cursor = chat_collection.find({"username": user["username"]})
    async for chat in cursor:
        # Convert MongoDB ObjectId to a string and remove '_id' field if desired
        chat["id"] = str(chat["_id"])
        del chat["_id"]
        chats.append(chat)
    return {"chats": chats}

# GET endpoint to retrieve a single chat by its id
@app.get("/chat_retrive/{chat_id}")
async def get_chat(chat_id: str, user: dict = Depends(get_current_user)):
    """
    Retrieve a single chat document by id for the current user.
    """
    try:
        oid = ObjectId(chat_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid chat id format")
    
    chat = await chat_collection.find_one({"_id": oid, "username": user["username"]})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat["id"] = str(chat["_id"])
    del chat["_id"]
    return chat