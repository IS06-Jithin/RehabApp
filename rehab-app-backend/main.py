"""
Rehab App Backend

This module serves as the backend for the Rehab App, providing API endpoints for
exercise management and real-time feedback.
"""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Rehab App Backend"}


@app.post("/start-exercise")
async def start_exercise():
    return {"message": "Exercise started"}
