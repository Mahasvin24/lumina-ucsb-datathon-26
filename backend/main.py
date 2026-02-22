from fastapi import FastAPI

from backend.app.api import router


app = FastAPI(title="Student Prediction API", version="0.1.0")
app.include_router(router)

