from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api import predict, transactions
from core.config import STATIC_DIR

class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response

app = FastAPI(title="Anti-fraud API")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")

app.include_router(predict.router)
app.include_router(transactions.router)
print('API is running...')