from fastapi import FastAPI, Request, Form, Depends, HTTPException
from retrieval import find_top_k_chunks, model, load_chunks_and_embeddings
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from fastapi.security import OAuth2PasswordRequestForm
from starlette.status import HTTP_302_FOUND
from pydantic import BaseModel
from model import generate_chat_response

from datetime import datetime, timezone
# from fastapi.staticfiles import StaticFiles  # Import StaticFiles

from bson import ObjectId
from db import chat_collection
from security.admin import authenticate_admin
from security.auth import create_access_token, get_current_user


# Initialize placeholders
main_chunks = None
embedding_cache = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan: load model and embeddings at startup."""
    global main_chunks, embedding_cache
    main_chunks, embedding_cache = load_chunks_and_embeddings()
    if not main_chunks or not embedding_cache:
        raise RuntimeError("Failed to load chunks or embeddings.")
    yield  # Application is running
    # Optional: Add teardown/cleanup logic here


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




class ChunkResponse(BaseModel):
    content: str
    metadata: dict = {}


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3  # Default number of retrieval results


@app.post("/chat")
async def chat_endpoint(request: ChatRequest, user: dict = Depends(get_current_user)):
    """
    Chat endpoint using real semantic retrieval and GPT-4 LLM generation.
    """
    
    try:
        
        if request.top_k > 0:
            print(request.top_k)
            relevant_chunks = find_top_k_chunks(request.query, top_k=request.top_k)
        else :
            relevant_chunks = []
        retrieval_results = [f"<strong>Retrieve_{i+1}</strong>: {chunk['content']}" for i, chunk in enumerate(relevant_chunks)]        
        # print(retrieval_results)

        # 🔥 Generate a real LLM-based response
        main_result = generate_chat_response(request.query, retrieval_results)

        print(main_result)

        # main_result = "Without Model: " + request.query
        chat_doc = {
            "username": user["username"],
            "query": request.query,
            "top_k": request.top_k,
            "retrieval_results": retrieval_results,
            "main_result": main_result,
            "timestamp": datetime.now(timezone.utc)
        }
        await chat_collection.insert_one(chat_doc)

        return {"retrieval": retrieval_results, "main": main_result}

    except Exception as e:
        print(f"Why error! {e}")
        
        raise HTTPException(status_code=500, detail=str(e))
    


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