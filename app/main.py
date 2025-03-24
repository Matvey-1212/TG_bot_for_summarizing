from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import config

app = FastAPI(title="ML Model API")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.APP_PORT,
        reload=config.DEBUG
    )
