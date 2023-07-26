import uvicorn
from fastapi import FastAPI
from model_processing import ModelsHandler
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor

app = FastAPI()

models_handler = ModelsHandler()

class Text(BaseModel):
    text: str

executor = ProcessPoolExecutor(max_workers=8)

@app.post("/process")
async def process(text: Text):
    futures = {model_name: executor.submit(model, text.text) for model_name, model in models_handler.models.items()}
    results = {model_name: future.result() for model_name, future in futures.items()}
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
