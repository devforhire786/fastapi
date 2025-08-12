# main.py
import datetime
import random
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from database import db_handler, get_db
from model import model_handler

# --- Application Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    db_handler.connect_to_db()
    model_handler.load_model()
    yield
    db_handler.close_db_connection()

app = FastAPI(lifespan=lifespan)

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# --- Pydantic Models ---
class MotionEvent(BaseModel):
    room: str
    motion_detected: bool
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

# --- API Endpoints ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint for real-time updates via WebSockets."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/motion/")
async def motion_event_endpoint(event: MotionEvent, db=Depends(get_db)):
    """Receives motion data, makes a prediction, and stores the result."""
    # 1. Prepare data for the AI model
    day_of_week = event.timestamp.weekday()
    time_of_day = event.timestamp.hour
    # Dummy past motion history for demonstration
    motion_history = [random.randint(0, 1) for _ in range(5)]

    input_data = np.array([[time_of_day, day_of_week] + motion_history])

    # 2. Run prediction in a thread pool to avoid blocking the server
    prediction_score = await run_in_threadpool(model_handler.predict, input_data)
    light_on = prediction_score > 0.5

    # 3. Save event and prediction to the database
    event_data = event.model_dump()
    event_data["light_on"] = light_on
    event_data["prediction_score"] = prediction_score
    db["motion_data"].insert_one(event_data)

    # 4. Broadcast the update to all connected WebSocket clients
    status = "ON" if light_on else "OFF"
    await manager.broadcast(f"Update for room '{event.room}': Light turned {status}")

    return {"room": event.room, "light_status": status}
@app.get("/")
def read_root():
    return {"message": "Welcome to the Smart Lighting API"}
