import uvicorn

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from load_data import load_hashedFilename, extract_photo_features, convert_data

import logging
import warnings

warnings.filterwarnings("ignore")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CONFIG_PATH = "config.yaml"


class LoanIdRequest(BaseModel):
    company_name: str
    loan_id: int


app = FastAPI()


@app.post("/get_features")
def get_photo_features(request: LoanIdRequest):
    """Pipeline for getting photo IDs from MySQL and calculating features"""
    logger.info(
        f"Starting request for loan_id: {request.loan_id}, "
        f"company: {request.company_name} in MySQL base"
    )
    df = load_hashedFilename(request.company_name, request.loan_id)

    dict_photo_features = extract_photo_features(request.company_name, df)
    logger.info(f"Created feature dictionary, photos count: {len(dict_photo_features)}")

    dict_photo_features_convert = convert_data(dict_photo_features)
    logger.info(f"Converted data types for JSON encoder")

    return JSONResponse(content=jsonable_encoder(dict_photo_features_convert))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.1", port=8080)
