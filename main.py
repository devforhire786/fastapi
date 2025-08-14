# main.py
import datetime
import random
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
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

class RoomPreferences(BaseModel):
    ai_enabled: bool = True
    auto_brightness: bool = True
    preferred_brightness: int = Field(default=80, ge=0, le=100)
    motion_sensitivity: str = Field(default="MEDIUM", regex="^(LOW|MEDIUM|HIGH)$")
    time_ranges: List[dict] = Field(default_factory=list)

class RoomCreate(BaseModel):
    name: str
    room_type: str
    preferences: Optional[RoomPreferences] = None

class RoomUpdate(BaseModel):
    name: Optional[str] = None
    room_type: Optional[str] = None
    preferences: Optional[RoomPreferences] = None

class Room(BaseModel):
    id: str
    name: str
    room_type: str
    preferences: RoomPreferences
    created_at: datetime.datetime
    updated_at: datetime.datetime

class LightStatus(BaseModel):
    status: str = Field(regex="^(ON|OFF)$")
    brightness: int = Field(ge=0, le=100)
    source: str = Field(regex="^(MANUAL|AI|SCHEDULE)$")

# --- API Endpoints ---

# Room Management Endpoints
@app.get("/rooms/", response_model=List[Room])
async def get_rooms(
    include_history: bool = False,
    include_stats: bool = False,
    db=Depends(get_db)
):
    """Retrieve all rooms with their current status."""
    try:
        rooms_collection = db["rooms"]
        rooms = list(rooms_collection.find())
        
        # Convert ObjectId to string for JSON serialization
        for room in rooms:
            room["id"] = str(room["_id"])
            del room["_id"]
        
        return rooms
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve rooms: {str(e)}")

@app.get("/rooms/{room_id}/", response_model=Room)
async def get_room(room_id: str, db=Depends(get_db)):
    """Get detailed information about a specific room."""
    try:
        from bson import ObjectId
        rooms_collection = db["rooms"]
        room = rooms_collection.find_one({"_id": ObjectId(room_id)})
        
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        
        room["id"] = str(room["_id"])
        del room["_id"]
        return room
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve room: {str(e)}")

@app.post("/rooms/", response_model=Room)
async def create_room(room_data: RoomCreate, db=Depends(get_db)):
    """Create a new room."""
    try:
        rooms_collection = db["rooms"]
        
        # Set default preferences if none provided
        if not room_data.preferences:
            room_data.preferences = RoomPreferences()
        
        room_doc = {
            "name": room_data.name,
            "room_type": room_data.room_type,
            "preferences": room_data.preferences.model_dump(),
            "created_at": datetime.datetime.now(),
            "updated_at": datetime.datetime.now()
        }
        
        result = rooms_collection.insert_one(room_doc)
        room_doc["id"] = str(result.inserted_id)
        del room_doc["_id"]
        
        return room_doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create room: {str(e)}")

@app.patch("/rooms/{room_id}/", response_model=Room)
async def update_room(room_id: str, room_update: RoomUpdate, db=Depends(get_db)):
    """Update room settings and preferences."""
    try:
        from bson import ObjectId
        rooms_collection = db["rooms"]
        
        # Prepare update data
        update_data = {}
        if room_update.name is not None:
            update_data["name"] = room_update.name
        if room_update.room_type is not None:
            update_data["room_type"] = room_update.room_type
        if room_update.preferences is not None:
            update_data["preferences"] = room_update.preferences.model_dump()
        
        update_data["updated_at"] = datetime.datetime.now()
        
        result = rooms_collection.update_one(
            {"_id": ObjectId(room_id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Return updated room
        updated_room = rooms_collection.find_one({"_id": ObjectId(room_id)})
        updated_room["id"] = str(updated_room["_id"])
        del updated_room["_id"]
        return updated_room
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update room: {str(e)}")

@app.delete("/rooms/{room_id}/")
async def delete_room(room_id: str, db=Depends(get_db)):
    """Remove a room from the system."""
    try:
        from bson import ObjectId
        rooms_collection = db["rooms"]
        
        result = rooms_collection.delete_one({"_id": ObjectId(room_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Room not found")
        
        return {"success": True, "message": "Room deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete room: {str(e)}")

# Light Control Endpoints
@app.get("/lights/")
async def get_all_lights(db=Depends(get_db)):
    """Get current status of all lights."""
    try:
        lights_collection = db["lights"]
        lights = list(lights_collection.find())
        
        # Convert ObjectId to string for JSON serialization
        for light in lights:
            light["id"] = str(light["_id"])
            del light["_id"]
        
        return {
            "success": True,
            "data": lights,
            "message": "Light status retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve light status: {str(e)}")

@app.get("/lights/{room_id}/")
async def get_room_light(room_id: str, db=Depends(get_db)):
    """Get current light status for a specific room."""
    try:
        from bson import ObjectId
        lights_collection = db["lights"]
        light = lights_collection.find_one({"room_id": room_id})
        
        if not light:
            # Create default light status if none exists
            light = {
                "room_id": room_id,
                "status": "OFF",
                "brightness": 0,
                "source": "MANUAL",
                "last_change": datetime.datetime.now()
            }
            lights_collection.insert_one(light)
            light["id"] = str(light["_id"])
            del light["_id"]
        else:
            light["id"] = str(light["_id"])
            del light["_id"]
        
        return {
            "success": True,
            "data": light,
            "message": "Light status retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve light status: {str(e)}")

@app.put("/lights/{room_id}/")
async def set_light_status(room_id: str, light_status: LightStatus, db=Depends(get_db)):
    """Set light to specific status."""
    try:
        from bson import ObjectId
        lights_collection = db["lights"]
        
        # Check if room exists
        rooms_collection = db["rooms"]
        room = rooms_collection.find_one({"_id": ObjectId(room_id)})
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Update or create light status
        light_data = {
            "room_id": room_id,
            "status": light_status.status,
            "brightness": light_status.brightness,
            "source": light_status.source,
            "last_change": datetime.datetime.now()
        }
        
        result = lights_collection.update_one(
            {"room_id": room_id},
            {"$set": light_data},
            upsert=True
        )
        
        # Broadcast update via WebSocket
        await manager.broadcast(f"Light update for room '{room_id}': {light_status.status} at {light_status.brightness}% brightness")
        
        return {
            "success": True,
            "data": light_data,
            "message": f"Light set to {light_status.status}",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set light status: {str(e)}")

@app.post("/lights/{room_id}/toggle/")
async def toggle_light(room_id: str, db=Depends(get_db)):
    """Toggle light on/off."""
    try:
        from bson import ObjectId
        lights_collection = db["lights"]
        
        # Get current light status
        current_light = lights_collection.find_one({"room_id": room_id})
        
        if not current_light:
            # Create default light if none exists
            new_status = "ON"
            new_brightness = 80
            light_data = {
                "room_id": room_id,
                "status": new_status,
                "brightness": new_brightness,
                "source": "MANUAL",
                "last_change": datetime.datetime.now()
            }
            lights_collection.insert_one(light_data)
        else:
            # Toggle existing light
            new_status = "OFF" if current_light["status"] == "ON" else "ON"
            new_brightness = current_light["brightness"] if new_status == "ON" else 0
            
            light_data = {
                "room_id": room_id,
                "status": new_status,
                "brightness": new_brightness,
                "source": "MANUAL",
                "last_change": datetime.datetime.now()
            }
            
            lights_collection.update_one(
                {"room_id": room_id},
                {"$set": light_data}
            )
        
        # Broadcast update via WebSocket
        await manager.broadcast(f"Light toggled for room '{room_id}': {new_status}")
        
        return {
            "success": True,
            "data": light_data,
            "message": f"Light toggled to {new_status}",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle light: {str(e)}")

@app.post("/lights/{room_id}/brightness/")
async def set_brightness(room_id: str, brightness_data: dict, db=Depends(get_db)):
    """Set specific brightness level."""
    try:
        brightness = brightness_data.get("brightness")
        if brightness is None or not isinstance(brightness, int) or brightness < 0 or brightness > 100:
            raise HTTPException(status_code=400, detail="Brightness must be an integer between 0 and 100")
        
        from bson import ObjectId
        lights_collection = db["lights"]
        
        # Update brightness
        light_data = {
            "room_id": room_id,
            "brightness": brightness,
            "status": "ON" if brightness > 0 else "OFF",
            "source": "MANUAL",
            "last_change": datetime.datetime.now()
        }
        
        lights_collection.update_one(
            {"room_id": room_id},
            {"$set": light_data},
            upsert=True
        )
        
        # Broadcast update via WebSocket
        await manager.broadcast(f"Brightness updated for room '{room_id}': {brightness}%")
        
        return {
            "success": True,
            "data": light_data,
            "message": f"Brightness set to {brightness}%",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set brightness: {str(e)}")

@app.post("/lights/{room_id}/dim/")
async def dim_light(room_id: str, dim_data: dict, db=Depends(get_db)):
    """Gradually dim light."""
    try:
        target_brightness = dim_data.get("target_brightness")
        duration_seconds = dim_data.get("duration_seconds", 30)
        
        if target_brightness is None or not isinstance(target_brightness, int) or target_brightness < 0 or target_brightness > 100:
            raise HTTPException(status_code=400, detail="Target brightness must be an integer between 0 and 100")
        
        if not isinstance(duration_seconds, int) or duration_seconds <= 0:
            raise HTTPException(status_code=400, detail="Duration must be a positive integer")
        
        from bson import ObjectId
        lights_collection = db["lights"]
        
        # Update light status
        light_data = {
            "room_id": room_id,
            "brightness": target_brightness,
            "status": "ON" if target_brightness > 0 else "OFF",
            "source": "MANUAL",
            "last_change": datetime.datetime.now()
        }
        
        lights_collection.update_one(
            {"room_id": room_id},
            {"$set": light_data},
            upsert=True
        )
        
        # Broadcast update via WebSocket
        await manager.broadcast(f"Light dimmed for room '{room_id}': {target_brightness}% over {duration_seconds} seconds")
        
        return {
            "success": True,
            "data": {
                **light_data,
                "dim_duration": duration_seconds
            },
            "message": f"Light dimmed to {target_brightness}% over {duration_seconds} seconds",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to dim light: {str(e)}")

# AI & Automation Endpoints
@app.get("/ai/status/")
async def get_ai_status(db=Depends(get_db)):
    """Get overall AI system status."""
    try:
        ai_collection = db["ai_status"]
        ai_status = ai_collection.find_one({"_id": "global"})
        
        if not ai_status:
            # Create default AI status
            ai_status = {
                "_id": "global",
                "enabled": True,
                "learning_mode": "ACTIVE",
                "accuracy": 0.85,
                "total_predictions": 0,
                "correct_predictions": 0,
                "last_updated": datetime.datetime.now()
            }
            ai_collection.insert_one(ai_status)
        
        # Calculate accuracy
        if ai_status["total_predictions"] > 0:
            accuracy = ai_status["correct_predictions"] / ai_status["total_predictions"]
        else:
            accuracy = 0.0
        
        return {
            "success": True,
            "data": {
                "enabled": ai_status["enabled"],
                "learning_mode": ai_status["learning_mode"],
                "accuracy": round(accuracy, 3),
                "total_predictions": ai_status["total_predictions"],
                "correct_predictions": ai_status["correct_predictions"],
                "last_updated": ai_status["last_updated"].isoformat()
            },
            "message": "AI status retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve AI status: {str(e)}")

@app.get("/ai/rooms/{room_id}/predictions/")
async def get_ai_predictions(room_id: str, db=Depends(get_db)):
    """Get AI predictions for a specific room."""
    try:
        from bson import ObjectId
        predictions_collection = db["ai_predictions"]
        
        # Get recent predictions for the room
        predictions = list(predictions_collection.find(
            {"room_id": room_id}
        ).sort("timestamp", -1).limit(10))
        
        # Convert ObjectId to string for JSON serialization
        for pred in predictions:
            pred["id"] = str(pred["_id"])
            del pred["_id"]
            pred["timestamp"] = pred["timestamp"].isoformat()
        
        return {
            "success": True,
            "data": predictions,
            "message": f"AI predictions retrieved for room {room_id}",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve AI predictions: {str(e)}")

@app.post("/ai/rooms/{room_id}/predict/")
async def request_ai_prediction(room_id: str, context: dict, db=Depends(get_db)):
    """Request AI prediction for a room."""
    try:
        from bson import ObjectId
        
        # Check if room exists
        rooms_collection = db["rooms"]
        room = rooms_collection.find_one({"_id": ObjectId(room_id)})
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Prepare input data for prediction
        current_time = datetime.datetime.now()
        day_of_week = current_time.weekday()
        time_of_day = current_time.hour
        
        # Get recent motion history
        motion_collection = db["motion_data"]
        recent_motion = list(motion_collection.find(
            {"room": room_id}
        ).sort("timestamp", -1).limit(5))
        
        motion_history = [1 if m["motion_detected"] else 0 for m in recent_motion]
        # Pad with zeros if less than 5 events
        while len(motion_history) < 5:
            motion_history.append(0)
        
        # Create input array for model
        input_data = np.array([[time_of_day, day_of_week] + motion_history])
        
        # Get prediction from model
        prediction_score = await run_in_threadpool(model_handler.predict, input_data)
        light_on = prediction_score > 0.5
        
        # Store prediction
        predictions_collection = db["ai_predictions"]
        prediction_data = {
            "room_id": room_id,
            "context": context,
            "prediction_score": float(prediction_score),
            "prediction": "ON" if light_on else "OFF",
            "timestamp": current_time,
            "input_features": {
                "time_of_day": time_of_day,
                "day_of_week": day_of_week,
                "motion_history": motion_history
            }
        }
        
        result = predictions_collection.insert_one(prediction_data)
        prediction_data["id"] = str(result.inserted_id)
        del prediction_data["_id"]
        
        # Update AI statistics
        ai_collection = db["ai_status"]
        ai_collection.update_one(
            {"_id": "global"},
            {
                "$inc": {"total_predictions": 1},
                "$set": {"last_updated": current_time}
            },
            upsert=True
        )
        
        # Broadcast prediction via WebSocket
        await manager.broadcast(f"AI prediction for room '{room_id}': Light should be {prediction_data['prediction']} (confidence: {prediction_score:.2f})")
        
        return {
            "success": True,
            "data": prediction_data,
            "message": f"AI prediction generated: Light should be {prediction_data['prediction']}",
            "timestamp": current_time.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate AI prediction: {str(e)}")

@app.put("/ai/global/")
async def update_global_ai(ai_config: dict, db=Depends(get_db)):
    """Enable/disable global AI control."""
    try:
        enabled = ai_config.get("enabled")
        if enabled is None or not isinstance(enabled, bool):
            raise HTTPException(status_code=400, detail="'enabled' field must be a boolean")
        
        ai_collection = db["ai_status"]
        ai_collection.update_one(
            {"_id": "global"},
            {
                "$set": {
                    "enabled": enabled,
                    "last_updated": datetime.datetime.now()
                }
            },
            upsert=True
        )
        
        # Broadcast update via WebSocket
        status = "enabled" if enabled else "disabled"
        await manager.broadcast(f"Global AI control {status}")
        
        return {
            "success": True,
            "data": {"enabled": enabled},
            "message": f"Global AI control {status}",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update global AI settings: {str(e)}")

@app.put("/ai/rooms/{room_id}/")
async def update_room_ai(room_id: str, ai_settings: dict, db=Depends(get_db)):
    """Configure AI settings for a specific room."""
    try:
        from bson import ObjectId
        
        # Check if room exists
        rooms_collection = db["rooms"]
        room = rooms_collection.find_one({"_id": ObjectId(room_id)})
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Update room preferences with AI settings
        update_data = {}
        if "ai_enabled" in ai_settings:
            update_data["preferences.ai_enabled"] = ai_settings["ai_enabled"]
        if "auto_brightness" in ai_settings:
            update_data["preferences.auto_brightness"] = ai_settings["auto_brightness"]
        if "motion_sensitivity" in ai_settings:
            update_data["preferences.motion_sensitivity"] = ai_settings["motion_sensitivity"]
        
        update_data["updated_at"] = datetime.datetime.now()
        
        result = rooms_collection.update_one(
            {"_id": ObjectId(room_id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Broadcast update via WebSocket
        await manager.broadcast(f"AI settings updated for room '{room_id}'")
        
        return {
            "success": True,
            "data": ai_settings,
            "message": "AI settings updated successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update AI settings: {str(e)}")

@app.get("/ai/learning/")
async def get_ai_learning(db=Depends(get_db)):
    """Get AI learning progress and accuracy."""
    try:
        # Get AI predictions with outcomes
        predictions_collection = db["ai_predictions"]
        motion_collection = db["motion_data"]
        
        # Get recent predictions
        recent_predictions = list(predictions_collection.find().sort("timestamp", -1).limit(100))
        
        # Calculate learning metrics
        total_predictions = len(recent_predictions)
        if total_predictions == 0:
            return {
                "success": True,
                "data": {
                    "total_predictions": 0,
                    "accuracy": 0.0,
                    "learning_progress": "No data yet",
                    "recent_improvements": []
                },
                "message": "No learning data available yet",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Calculate accuracy (this is a simplified version)
        # In a real system, you'd compare predictions with actual outcomes
        correct_predictions = sum(1 for p in recent_predictions if p.get("prediction_score", 0) > 0.5)
        accuracy = correct_predictions / total_predictions
        
        # Get learning progress over time
        learning_data = []
        for pred in recent_predictions[-10:]:  # Last 10 predictions
            learning_data.append({
                "timestamp": pred["timestamp"].isoformat(),
                "prediction_score": pred["prediction_score"],
                "confidence": "HIGH" if pred["prediction_score"] > 0.7 else "MEDIUM" if pred["prediction_score"] > 0.5 else "LOW"
            })
        
        return {
            "success": True,
            "data": {
                "total_predictions": total_predictions,
                "accuracy": round(accuracy, 3),
                "learning_progress": "Active" if accuracy > 0.6 else "Learning",
                "recent_improvements": learning_data
            },
            "message": "AI learning data retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve AI learning data: {str(e)}")

# Motion Detection Endpoints
@app.get("/motion/rooms/{room_id}/")
async def get_room_motion_status(room_id: str, db=Depends(get_db)):
    """Get current motion status for a room."""
    try:
        from bson import ObjectId
        
        # Check if room exists
        rooms_collection = db["rooms"]
        room = rooms_collection.find_one({"_id": ObjectId(room_id)})
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Get latest motion event for the room
        motion_collection = db["motion_data"]
        latest_motion = motion_collection.find_one(
            {"room": room_id},
            sort=[("timestamp", -1)]
        )
        
        if not latest_motion:
            motion_status = {
                "motion_detected": False,
                "last_motion_time": None,
                "confidence": 0.0
            }
        else:
            motion_status = {
                "motion_detected": latest_motion["motion_detected"],
                "last_motion_time": latest_motion["timestamp"].isoformat(),
                "confidence": latest_motion.get("confidence", 0.8)
            }
        
        return {
            "success": True,
            "data": motion_status,
            "message": f"Motion status retrieved for room {room_id}",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve motion status: {str(e)}")

@app.put("/motion/rooms/{room_id}/sensitivity/")
async def update_motion_sensitivity(room_id: str, sensitivity_data: dict, db=Depends(get_db)):
    """Adjust motion detection sensitivity."""
    try:
        from bson import ObjectId
        
        sensitivity = sensitivity_data.get("sensitivity")
        if sensitivity not in ["LOW", "MEDIUM", "HIGH"]:
            raise HTTPException(status_code=400, detail="Sensitivity must be LOW, MEDIUM, or HIGH")
        
        # Check if room exists
        rooms_collection = db["rooms"]
        room = rooms_collection.find_one({"_id": ObjectId(room_id)})
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Update room preferences with new sensitivity
        result = rooms_collection.update_one(
            {"_id": ObjectId(room_id)},
            {
                "$set": {
                    "preferences.motion_sensitivity": sensitivity,
                    "updated_at": datetime.datetime.now()
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Broadcast update via WebSocket
        await manager.broadcast(f"Motion sensitivity updated for room '{room_id}': {sensitivity}")
        
        return {
            "success": True,
            "data": {"sensitivity": sensitivity},
            "message": f"Motion sensitivity updated to {sensitivity}",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update motion sensitivity: {str(e)}")

@app.get("/motion/calibration/")
async def get_motion_calibration(db=Depends(get_db)):
    """Get motion sensor calibration status."""
    try:
        # Get calibration data from motion sensors
        motion_collection = db["motion_data"]
        
        # Get recent motion events to analyze sensor health
        recent_events = list(motion_collection.find().sort("timestamp", -1).limit(100))
        
        if not recent_events:
            calibration_data = {
                "sensor_health": "UNKNOWN",
                "calibration_status": "NO_DATA",
                "total_events": 0,
                "sensor_issues": []
            }
        else:
            # Analyze sensor health based on event patterns
            total_events = len(recent_events)
            motion_events = sum(1 for e in recent_events if e["motion_detected"])
            motion_ratio = motion_events / total_events if total_events > 0 else 0
            
            # Simple health assessment
            if total_events < 10:
                sensor_health = "INSUFFICIENT_DATA"
            elif motion_ratio > 0.8 or motion_ratio < 0.2:
                sensor_health = "POTENTIAL_ISSUE"
            else:
                sensor_health = "HEALTHY"
            
            calibration_data = {
                "sensor_health": sensor_health,
                "calibration_status": "CALIBRATED" if sensor_health == "HEALTHY" else "NEEDS_CALIBRATION",
                "total_events": total_events,
                "motion_ratio": round(motion_ratio, 3),
                "sensor_issues": []
            }
        
        return {
            "success": True,
            "data": calibration_data,
            "message": "Motion calibration status retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve motion calibration: {str(e)}")

# Scheduling & Time Management Endpoints
@app.get("/schedule/")
async def get_all_schedules(db=Depends(get_db)):
    """Get all scheduled lighting events."""
    try:
        schedule_collection = db["schedules"]
        schedules = list(schedule_collection.find().sort("created_at", -1))
        
        # Convert ObjectId to string for JSON serialization
        for schedule in schedules:
            schedule["id"] = str(schedule["_id"])
            del schedule["_id"]
            schedule["created_at"] = schedule["created_at"].isoformat()
            schedule["updated_at"] = schedule["updated_at"].isoformat()
        
        return {
            "success": True,
            "data": schedules,
            "message": "Schedules retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve schedules: {str(e)}")

@app.post("/schedule/")
async def create_schedule(schedule_data: dict, db=Depends(get_db)):
    """Create a new lighting schedule."""
    try:
        # Validate required fields
        required_fields = ["room_id", "time", "action", "repeat"]
        for field in required_fields:
            if field not in schedule_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Validate action
        if schedule_data["action"] not in ["ON", "OFF", "BRIGHTNESS"]:
            raise HTTPException(status_code=400, detail="Action must be ON, OFF, or BRIGHTNESS")
        
        # Validate repeat
        if schedule_data["repeat"] not in ["ONCE", "DAILY", "WEEKLY", "MONTHLY"]:
            raise HTTPException(status_code=400, detail="Repeat must be ONCE, DAILY, WEEKLY, or MONTHLY")
        
        # Check if room exists
        from bson import ObjectId
        rooms_collection = db["rooms"]
        room = rooms_collection.find_one({"_id": ObjectId(schedule_data["room_id"])})
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Create schedule document
        schedule_doc = {
            "room_id": schedule_data["room_id"],
            "time": schedule_data["time"],
            "action": schedule_data["action"],
            "repeat": schedule_data["repeat"],
            "enabled": schedule_data.get("enabled", True),
            "brightness": schedule_data.get("brightness", 80),
            "created_at": datetime.datetime.now(),
            "updated_at": datetime.datetime.now()
        }
        
        schedule_collection = db["schedules"]
        result = schedule_collection.insert_one(schedule_doc)
        schedule_doc["id"] = str(result.inserted_id)
        del schedule_doc["_id"]
        
        # Broadcast new schedule via WebSocket
        await manager.broadcast(f"New schedule created for room '{schedule_data['room_id']}': {schedule_data['action']} at {schedule_data['time']}")
        
        return {
            "success": True,
            "data": schedule_doc,
            "message": "Schedule created successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create schedule: {str(e)}")

@app.put("/schedule/{schedule_id}/")
async def update_schedule(schedule_id: str, schedule_update: dict, db=Depends(get_db)):
    """Update existing schedule."""
    try:
        from bson import ObjectId
        schedule_collection = db["schedules"]
        
        # Check if schedule exists
        existing_schedule = schedule_collection.find_one({"_id": ObjectId(schedule_id)})
        if not existing_schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        # Prepare update data
        update_data = {}
        allowed_fields = ["time", "action", "repeat", "enabled", "brightness"]
        
        for field in allowed_fields:
            if field in schedule_update:
                update_data[field] = schedule_update[field]
        
        update_data["updated_at"] = datetime.datetime.now()
        
        # Validate action if provided
        if "action" in update_data and update_data["action"] not in ["ON", "OFF", "BRIGHTNESS"]:
            raise HTTPException(status_code=400, detail="Action must be ON, OFF, or BRIGHTNESS")
        
        # Validate repeat if provided
        if "repeat" in update_data and update_data["repeat"] not in ["ONCE", "DAILY", "WEEKLY", "MONTHLY"]:
            raise HTTPException(status_code=400, detail="Repeat must be ONCE, DAILY, WEEKLY, or MONTHLY")
        
        # Update schedule
        result = schedule_collection.update_one(
            {"_id": ObjectId(schedule_id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        # Return updated schedule
        updated_schedule = schedule_collection.find_one({"_id": ObjectId(schedule_id)})
        updated_schedule["id"] = str(updated_schedule["_id"])
        del updated_schedule["_id"]
        updated_schedule["created_at"] = updated_schedule["created_at"].isoformat()
        updated_schedule["updated_at"] = updated_schedule["updated_at"].isoformat()
        
        # Broadcast update via WebSocket
        await manager.broadcast(f"Schedule {schedule_id} updated")
        
        return {
            "success": True,
            "data": updated_schedule,
            "message": "Schedule updated successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update schedule: {str(e)}")

@app.delete("/schedule/{schedule_id}/")
async def delete_schedule(schedule_id: str, db=Depends(get_db)):
    """Remove a schedule."""
    try:
        from bson import ObjectId
        schedule_collection = db["schedules"]
        
        result = schedule_collection.delete_one({"_id": ObjectId(schedule_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        # Broadcast deletion via WebSocket
        await manager.broadcast(f"Schedule {schedule_id} deleted")
        
        return {
            "success": True,
            "message": "Schedule deleted successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete schedule: {str(e)}")

@app.get("/schedule/rooms/{room_id}/")
async def get_room_schedules(room_id: str, db=Depends(get_db)):
    """Get schedules for a specific room."""
    try:
        from bson import ObjectId
        
        # Check if room exists
        rooms_collection = db["rooms"]
        room = rooms_collection.find_one({"_id": ObjectId(room_id)})
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Get schedules for the room
        schedule_collection = db["schedules"]
        schedules = list(schedule_collection.find({"room_id": room_id}).sort("created_at", -1))
        
        # Convert ObjectId to string for JSON serialization
        for schedule in schedules:
            schedule["id"] = str(schedule["_id"])
            del schedule["_id"]
            schedule["created_at"] = schedule["created_at"].isoformat()
            schedule["updated_at"] = schedule["updated_at"].isoformat()
        
        return {
            "success": True,
            "data": schedules,
            "message": f"Schedules retrieved for room {room_id}",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve room schedules: {str(e)}")

# Analytics & History Endpoints
@app.get("/analytics/overview/")
async def get_analytics_overview(db=Depends(get_db)):
    """Get overall system analytics."""
    try:
        # Get data from various collections
        rooms_collection = db["rooms"]
        lights_collection = db["lights"]
        motion_collection = db["motion_data"]
        ai_collection = db["ai_status"]
        
        # Calculate basic statistics
        total_rooms = rooms_collection.count_documents({})
        total_lights = lights_collection.count_documents({})
        total_motion_events = motion_collection.count_documents({})
        
        # Get AI accuracy
        ai_status = ai_collection.find_one({"_id": "global"})
        ai_accuracy = ai_status.get("accuracy", 0.0) if ai_status else 0.0
        
        # Calculate motion patterns (last 24 hours)
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        recent_motion = motion_collection.count_documents({"timestamp": {"$gte": yesterday}})
        
        # Calculate energy usage (simplified - based on light status)
        active_lights = lights_collection.count_documents({"status": "ON"})
        energy_usage = active_lights * 0.1  # Simplified calculation
        
        analytics_data = {
            "total_rooms": total_rooms,
            "total_lights": total_lights,
            "total_motion_events": total_motion_events,
            "ai_accuracy": round(ai_accuracy, 3),
            "recent_motion_events": recent_motion,
            "active_lights": active_lights,
            "estimated_energy_usage_kwh": round(energy_usage, 2),
            "system_health": "HEALTHY" if ai_accuracy > 0.7 else "NEEDS_ATTENTION"
        }
        
        return {
            "success": True,
            "data": analytics_data,
            "message": "Analytics overview retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analytics overview: {str(e)}")

@app.get("/analytics/rooms/{room_id}/")
async def get_room_analytics(room_id: str, db=Depends(get_db)):
    """Get analytics for a specific room."""
    try:
        from bson import ObjectId
        
        # Check if room exists
        rooms_collection = db["rooms"]
        room = rooms_collection.find_one({"_id": ObjectId(room_id)})
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Get room-specific data
        lights_collection = db["lights"]
        motion_collection = db["motion_data"]
        predictions_collection = db["ai_predictions"]
        
        # Get light status
        light_status = lights_collection.find_one({"room_id": room_id})
        current_brightness = light_status.get("brightness", 0) if light_status else 0
        light_on_time = light_status.get("last_change", datetime.datetime.now()) if light_status else datetime.datetime.now()
        
        # Get motion statistics (last 7 days)
        week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
        motion_events = list(motion_collection.find({
            "room": room_id,
            "timestamp": {"$gte": week_ago}
        }))
        
        motion_count = len(motion_events)
        motion_detected_count = sum(1 for e in motion_events if e["motion_detected"])
        motion_ratio = motion_detected_count / motion_count if motion_count > 0 else 0
        
        # Get AI predictions for the room
        room_predictions = list(predictions_collection.find({"room_id": room_id}).sort("timestamp", -1).limit(10))
        prediction_accuracy = 0.0
        if room_predictions:
            # Simplified accuracy calculation
            prediction_accuracy = sum(1 for p in room_predictions if p.get("prediction_score", 0) > 0.5) / len(room_predictions)
        
        room_analytics = {
            "room_id": room_id,
            "room_name": room.get("name", "Unknown"),
            "current_brightness": current_brightness,
            "light_last_changed": light_on_time.isoformat() if isinstance(light_on_time, datetime.datetime) else str(light_on_time),
            "motion_events_7_days": motion_count,
            "motion_detection_ratio": round(motion_ratio, 3),
            "ai_prediction_accuracy": round(prediction_accuracy, 3),
            "efficiency_score": round((motion_ratio + prediction_accuracy) / 2, 3)
        }
        
        return {
            "success": True,
            "data": room_analytics,
            "message": f"Room analytics retrieved for {room_id}",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve room analytics: {str(e)}")

@app.get("/history/lights/")
async def get_light_history(
    room_id: Optional[str] = None,
    limit: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db=Depends(get_db)
):
    """Get light change history."""
    try:
        lights_collection = db["lights"]
        
        # Build query
        query = {}
        if room_id:
            query["room_id"] = room_id
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                try:
                    start_dt = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    date_query["$gte"] = start_dt
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO format (YYYY-MM-DD)")
            
            if end_date:
                try:
                    end_dt = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    date_query["$lte"] = end_dt
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO format (YYYY-MM-DD)")
            
            query["last_change"] = date_query
        
        # Get light history
        light_history = list(lights_collection.find(query).sort("last_change", -1).limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for light in light_history:
            light["id"] = str(light["_id"])
            del light["_id"]
            if "last_change" in light and isinstance(light["last_change"], datetime.datetime):
                light["last_change"] = light["last_change"].isoformat()
        
        return {
            "success": True,
            "data": light_history,
            "message": "Light history retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat(),
            "pagination": {
                "limit": limit,
                "total": len(light_history),
                "has_more": len(light_history) == limit
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve light history: {str(e)}")

@app.get("/history/motion/")
async def get_motion_history(
    room_id: Optional[str] = None,
    limit: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db=Depends(get_db)
):
    """Get motion event history."""
    try:
        motion_collection = db["motion_data"]
        
        # Build query
        query = {}
        if room_id:
            query["room"] = room_id
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                try:
                    start_dt = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    date_query["$gte"] = start_dt
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO format (YYYY-MM-DD)")
            
            if end_date:
                try:
                    end_dt = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    date_query["$lte"] = end_dt
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO format (YYYY-MM-DD)")
            
            query["timestamp"] = date_query
        
        # Get motion history
        motion_history = list(motion_collection.find(query).sort("timestamp", -1).limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for motion in motion_history:
            motion["id"] = str(motion["_id"])
            del motion["_id"]
            if "timestamp" in motion and isinstance(motion["timestamp"], datetime.datetime):
                motion["timestamp"] = motion["timestamp"].isoformat()
        
        return {
            "success": True,
            "data": motion_history,
            "message": "Motion history retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat(),
            "pagination": {
                "limit": limit,
                "total": len(motion_history),
                "has_more": len(motion_history) == limit
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve motion history: {str(e)}")

@app.get("/history/ai/")
async def get_ai_history(db=Depends(get_db)):
    """Get AI decision history."""
    try:
        predictions_collection = db["ai_predictions"]
        
        # Get all AI predictions
        ai_history = list(predictions_collection.find().sort("timestamp", -1).limit(100))
        
        # Convert ObjectId to string for JSON serialization
        for prediction in ai_history:
            prediction["id"] = str(prediction["_id"])
            del prediction["_id"]
            if "timestamp" in prediction and isinstance(prediction["timestamp"], datetime.datetime):
                prediction["timestamp"] = prediction["timestamp"].isoformat()
        
        return {
            "success": True,
            "data": ai_history,
            "message": "AI decision history retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat(),
            "pagination": {
                "limit": 100,
                "total": len(ai_history),
                "has_more": len(ai_history) == 100
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve AI history: {str(e)}")

@app.get("/analytics/energy/")
async def get_energy_analytics(db=Depends(get_db)):
    """Get energy consumption analytics."""
    try:
        lights_collection = db["lights"]
        rooms_collection = db["rooms"]
        
        # Get all lights
        all_lights = list(lights_collection.find())
        
        # Calculate energy usage by room
        energy_by_room = {}
        total_energy = 0.0
        
        for light in all_lights:
            room_id = light.get("room_id")
            if room_id:
                # Get room name
                room = rooms_collection.find_one({"_id": room_id})
                room_name = room.get("name", "Unknown") if room else "Unknown"
                
                # Calculate energy (simplified - based on brightness and time)
                brightness = light.get("brightness", 0)
                status = light.get("status", "OFF")
                
                if status == "ON" and brightness > 0:
                    # Simplified energy calculation: brightness * 0.001 kWh per hour
                    energy = (brightness / 100) * 0.001
                    total_energy += energy
                    
                    if room_name not in energy_by_room:
                        energy_by_room[room_name] = 0.0
                    energy_by_room[room_name] += energy
        
        # Calculate cost (assuming $0.12 per kWh)
        cost_per_kwh = 0.12
        total_cost = total_energy * cost_per_kwh
        
        energy_data = {
            "total_energy_kwh": round(total_energy, 3),
            "total_cost_usd": round(total_cost, 2),
            "energy_by_room": {room: round(energy, 3) for room, energy in energy_by_room.items()},
            "cost_per_kwh": cost_per_kwh,
            "calculation_method": "Simplified based on brightness and status"
        }
        
        return {
            "success": True,
            "data": energy_data,
            "message": "Energy analytics retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve energy analytics: {str(e)}")

# System Configuration Endpoints
@app.get("/system/status/")
async def get_system_status(db=Depends(get_db)):
    """Get overall system health."""
    try:
        # Check database connection
        try:
            db.command("ping")
            db_status = "CONNECTED"
        except Exception:
            db_status = "DISCONNECTED"
        
        # Check model status
        model_status = "LOADED" if model_handler.model is not None else "NOT_LOADED"
        
        # Get system statistics
        rooms_collection = db["rooms"]
        lights_collection = db["lights"]
        motion_collection = db["motion_data"]
        
        total_rooms = rooms_collection.count_documents({})
        total_lights = lights_collection.count_documents({})
        total_motion_events = motion_collection.count_documents({})
        
        # Check for recent errors (last 24 hours)
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        recent_errors = 0  # In a real system, you'd have an error log collection
        
        system_health = "HEALTHY"
        if db_status == "DISCONNECTED" or model_status == "NOT_LOADED":
            system_health = "CRITICAL"
        elif recent_errors > 10:
            system_health = "WARNING"
        
        system_status = {
            "overall_health": system_health,
            "database": db_status,
            "ml_model": model_status,
            "total_rooms": total_rooms,
            "total_lights": total_lights,
            "total_motion_events": total_motion_events,
            "recent_errors": recent_errors,
            "uptime": "Unknown",  # In a real system, track actual uptime
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "data": system_status,
            "message": "System status retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system status: {str(e)}")

@app.get("/system/devices/")
async def get_system_devices(db=Depends(get_db)):
    """Get connected IoT devices."""
    try:
        devices_collection = db["devices"]
        devices = list(devices_collection.find())
        
        # Convert ObjectId to string for JSON serialization
        for device in devices:
            device["id"] = str(device["_id"])
            del device["_id"]
            if "last_seen" in device and isinstance(device["last_seen"], datetime.datetime):
                device["last_seen"] = device["last_seen"].isoformat()
        
        return {
            "success": True,
            "data": devices,
            "message": "System devices retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system devices: {str(e)}")

@app.post("/system/devices/")
async def add_system_device(device_data: dict, db=Depends(get_db)):
    """Add new IoT device."""
    try:
        # Validate required fields
        required_fields = ["name", "type", "ip_address"]
        for field in required_fields:
            if field not in device_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Validate device type
        valid_types = ["LIGHT", "MOTION_SENSOR", "CONTROLLER", "GATEWAY"]
        if device_data["type"] not in valid_types:
            raise HTTPException(status_code=400, detail=f"Device type must be one of: {', '.join(valid_types)}")
        
        # Create device document
        device_doc = {
            "name": device_data["name"],
            "type": device_data["type"],
            "ip_address": device_data["ip_address"],
            "status": "ONLINE",
            "last_seen": datetime.datetime.now(),
            "created_at": datetime.datetime.now(),
            "updated_at": datetime.datetime.now()
        }
        
        # Add optional fields
        if "room_id" in device_data:
            device_doc["room_id"] = device_data["room_id"]
        if "capabilities" in device_data:
            device_doc["capabilities"] = device_data["capabilities"]
        
        devices_collection = db["devices"]
        result = devices_collection.insert_one(device_doc)
        device_doc["id"] = str(result.inserted_id)
        del device_doc["_id"]
        
        # Broadcast new device via WebSocket
        await manager.broadcast(f"New device added: {device_data['name']} ({device_data['type']})")
        
        return {
            "success": True,
            "data": device_doc,
            "message": "Device added successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add device: {str(e)}")

@app.put("/system/devices/{device_id}/")
async def update_system_device(device_id: str, device_update: dict, db=Depends(get_db)):
    """Update device settings."""
    try:
        from bson import ObjectId
        devices_collection = db["devices"]
        
        # Check if device exists
        existing_device = devices_collection.find_one({"_id": ObjectId(device_id)})
        if not existing_device:
            raise HTTPException(status_code=404, detail="Device not found")
        
        # Prepare update data
        update_data = {}
        allowed_fields = ["name", "ip_address", "status", "capabilities", "room_id"]
        
        for field in allowed_fields:
            if field in device_update:
                update_data[field] = device_update[field]
        
        update_data["updated_at"] = datetime.datetime.now()
        
        # Update device
        result = devices_collection.update_one(
            {"_id": ObjectId(device_id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Device not found")
        
        # Return updated device
        updated_device = devices_collection.find_one({"_id": ObjectId(device_id)})
        updated_device["id"] = str(updated_device["_id"])
        del updated_device["_id"]
        updated_device["created_at"] = updated_device["created_at"].isoformat()
        updated_device["updated_at"] = updated_device["updated_at"].isoformat()
        if "last_seen" in updated_device and isinstance(updated_device["last_seen"], datetime.datetime):
            updated_device["last_seen"] = updated_device["last_seen"].isoformat()
        
        # Broadcast update via WebSocket
        await manager.broadcast(f"Device {device_id} updated")
        
        return {
            "success": True,
            "data": updated_device,
            "message": "Device updated successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update device: {str(e)}")

@app.get("/system/logs/")
async def get_system_logs(
    level: Optional[str] = None,
    limit: int = 100,
    start_date: Optional[str] = None,
    db=Depends(get_db)
):
    """Get system logs."""
    try:
        logs_collection = db["system_logs"]
        
        # Build query
        query = {}
        if level:
            if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise HTTPException(status_code=400, detail="Invalid log level")
            query["level"] = level
        
        if start_date:
            try:
                start_dt = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                query["timestamp"] = {"$gte": start_dt}
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO format (YYYY-MM-DD)")
        
        # Get logs
        logs = list(logs_collection.find(query).sort("timestamp", -1).limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for log in logs:
            log["id"] = str(log["_id"])
            del log["_id"]
            if "timestamp" in log and isinstance(log["timestamp"], datetime.datetime):
                log["timestamp"] = log["timestamp"].isoformat()
        
        return {
            "success": True,
            "data": logs,
            "message": "System logs retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat(),
            "pagination": {
                "limit": limit,
                "total": len(logs),
                "has_more": len(logs) == limit
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system logs: {str(e)}")

# Mobile App Specific Endpoints
@app.get("/mobile/status/")
async def get_mobile_status():
    """Get mobile app specific status."""
    try:
        mobile_status = {
            "app_version": "1.0.0",
            "required_updates": False,
            "maintenance_status": "NONE",
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "data": mobile_status,
            "message": "Mobile app status retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve mobile status: {str(e)}")

@app.post("/mobile/feedback/")
async def submit_mobile_feedback(feedback_data: dict, db=Depends(get_db)):
    """Submit app feedback."""
    try:
        # Validate required fields
        if "rating" not in feedback_data or "comment" not in feedback_data:
            raise HTTPException(status_code=400, detail="Rating and comment are required")
        
        rating = feedback_data["rating"]
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be an integer between 1 and 5")
        
        # Store feedback
        feedback_doc = {
            "rating": rating,
            "comment": feedback_data["comment"],
            "user_id": feedback_data.get("user_id", "anonymous"),
            "timestamp": datetime.datetime.now()
        }
        
        feedback_collection = db["mobile_feedback"]
        result = feedback_collection.insert_one(feedback_doc)
        feedback_doc["id"] = str(result.inserted_id)
        del feedback_doc["_id"]
        
        return {
            "success": True,
            "data": feedback_doc,
            "message": "Feedback submitted successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@app.get("/mobile/help/")
async def get_mobile_help():
    """Get help content and FAQs."""
    try:
        help_content = {
            "faqs": [
                {
                    "question": "How do I control my lights?",
                    "answer": "Use the light control endpoints to turn lights on/off, adjust brightness, or set schedules."
                },
                {
                    "question": "How does the AI work?",
                    "answer": "The AI learns from your motion patterns and automatically adjusts lighting based on your habits."
                },
                {
                    "question": "Can I set up schedules?",
                    "answer": "Yes! Use the scheduling endpoints to create automated lighting schedules."
                }
            ],
            "troubleshooting": [
                "Check if your device is connected to the network",
                "Ensure the room exists in the system",
                "Verify motion sensors are working properly"
            ]
        }
        
        return {
            "success": True,
            "data": help_content,
            "message": "Help content retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve help content: {str(e)}")

# Security & Permissions Endpoints
@app.get("/permissions/")
async def get_permissions(db=Depends(get_db)):
    """Get current permissions."""
    try:
        # In a single-user system, return basic permissions
        permissions = {
            "available_actions": [
                "READ_ROOMS", "WRITE_ROOMS", "READ_LIGHTS", "WRITE_LIGHTS",
                "READ_ANALYTICS", "READ_SYSTEM", "WRITE_SYSTEM"
            ],
            "room_access": "ALL_ROOMS",
            "permission_level": "ADMIN"
        }
        
        return {
            "success": True,
            "data": permissions,
            "message": "Permissions retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve permissions: {str(e)}")

# Data Export & Import Endpoints
@app.get("/export/rooms/{room_id}/data/")
async def export_room_data(
    room_id: str,
    format: str = "JSON",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db=Depends(get_db)
):
    """Export room data."""
    try:
        from bson import ObjectId
        
        # Check if room exists
        rooms_collection = db["rooms"]
        room = rooms_collection.find_one({"_id": ObjectId(room_id)})
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        
        if format not in ["JSON", "CSV"]:
            raise HTTPException(status_code=400, detail="Format must be JSON or CSV")
        
        # Get room data
        room_data = {
            "room_info": room,
            "light_history": [],
            "motion_history": [],
            "ai_predictions": []
        }
        
        # Get light history
        lights_collection = db["lights"]
        light_history = list(lights_collection.find({"room_id": room_id}))
        room_data["light_history"] = light_history
        
        # Get motion history
        motion_collection = db["motion_data"]
        motion_history = list(motion_collection.find({"room": room_id}))
        room_data["motion_history"] = motion_history
        
        # Get AI predictions
        predictions_collection = db["ai_predictions"]
        ai_predictions = list(predictions_collection.find({"room_id": room_id}))
        room_data["ai_predictions"] = ai_predictions
        
        return {
            "success": True,
            "data": room_data,
            "message": f"Room data exported successfully in {format} format",
            "timestamp": datetime.datetime.now().isoformat(),
            "export_format": format
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export room data: {str(e)}")

# Testing & Development Endpoints
@app.post("/test/motion/")
async def test_motion_event(test_data: dict, db=Depends(get_db)):
    """Simulate motion event for testing."""
    try:
        # Validate test data
        if "room" not in test_data or "motion_detected" not in test_data:
            raise HTTPException(status_code=400, detail="Room and motion_detected are required")
        
        # Create test motion event
        test_event = MotionEvent(
            room=test_data["room"],
            motion_detected=test_data["motion_detected"],
            timestamp=datetime.datetime.now()
        )
        
        # Process the test event (same as real motion endpoint)
        day_of_week = test_event.timestamp.weekday()
        time_of_day = test_event.timestamp.hour
        motion_history = [random.randint(0, 1) for _ in range(5)]
        
        input_data = np.array([[time_of_day, day_of_week] + motion_history])
        prediction_score = await run_in_threadpool(model_handler.predict, input_data)
        light_on = prediction_score > 0.5
        
        # Store test event
        event_data = test_event.model_dump()
        event_data["light_on"] = light_on
        event_data["prediction_score"] = prediction_score
        event_data["is_test"] = True
        
        motion_collection = db["motion_data"]
        motion_collection.insert_one(event_data)
        
        # Broadcast test event via WebSocket
        status = "ON" if light_on else "OFF"
        await manager.broadcast(f"TEST: Motion event for room '{test_event.room}': Light turned {status}")
        
        return {
            "success": True,
            "data": {
                "room": test_event.room,
                "motion_detected": test_event.motion_detected,
                "light_status": status,
                "prediction_score": float(prediction_score),
                "is_test": True
            },
            "message": "Test motion event processed successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process test motion event: {str(e)}")

@app.post("/test/lights/")
async def test_light_change(test_data: dict, db=Depends(get_db)):
    """Simulate light change for testing."""
    try:
        # Validate test data
        if "room_id" not in test_data or "status" not in test_data:
            raise HTTPException(status_code=400, detail="room_id and status are required")
        
        # Create test light change
        light_data = {
            "room_id": test_data["room_id"],
            "status": test_data["status"],
            "brightness": test_data.get("brightness", 80),
            "source": "TEST",
            "last_change": datetime.datetime.now(),
            "is_test": True
        }
        
        # Store test light change
        lights_collection = db["lights"]
        lights_collection.update_one(
            {"room_id": test_data["room_id"]},
            {"$set": light_data},
            upsert=True
        )
        
        # Broadcast test light change via WebSocket
        await manager.broadcast(f"TEST: Light changed for room '{test_data['room_id']}': {test_data['status']}")
        
        return {
            "success": True,
            "data": light_data,
            "message": "Test light change processed successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process test light change: {str(e)}")

@app.get("/test/status/")
async def get_test_status():
    """Get test environment status."""
    try:
        test_status = {
            "test_mode": True,
            "test_endpoints_available": True,
            "test_data_cleanup": "MANUAL",
            "last_test_run": datetime.datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "data": test_status,
            "message": "Test environment status retrieved successfully",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve test status: {str(e)}")

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
    try:
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

        return {
            "success": True,
            "data": {
                "room": event.room,
                "light_status": status,
                "prediction_score": float(prediction_score),
                "timestamp": event.timestamp.isoformat()
            },
            "message": f"Motion event processed successfully. Light turned {status}",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process motion event: {str(e)}")
@app.get("/")
def read_root():
    return {"message": "Welcome to the Smart Lighting API"}
