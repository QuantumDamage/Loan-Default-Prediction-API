from typing import List

import numpy as np
import onnxruntime as rt
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist, field_validator

app = FastAPI()


class Record(BaseModel):
    LIMIT_BAL: int
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: int
    BILL_AMT2: int
    BILL_AMT3: int
    BILL_AMT4: int
    BILL_AMT5: int
    BILL_AMT6: int
    PAY_AMT1: int
    PAY_AMT2: int
    PAY_AMT3: int
    PAY_AMT4: int
    PAY_AMT5: int
    PAY_AMT6: int


class InputData(BaseModel):
    records: List[Record]

    @field_validator("records")
    def check_length(cls, v):
        if len(v) < 1:
            raise ValueError("At least one record must be provided")
        return v

categorical_columns = [1, 2, 3, 5, 6, 7, 8, 9, 10]
continuous_columns = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

one_hot_encoder_onnx = rt.InferenceSession("src/models/one_hot_encoder.onnx")
one_hot_encoder_input_name = one_hot_encoder_onnx.get_inputs()[0].name
standard_scaler_onnx = rt.InferenceSession("src/models/standard_scaler.onnx")
standard_scaler_input_name = standard_scaler_onnx.get_inputs()[0].name

@app.post("/predict/")
def predict(input_data: InputData):
    data_list = [list(record.model_dump().values()) for record in input_data.records]
    data_array = np.array(data_list)

    if not np.issubdtype(data_array.dtype, np.int64):
        raise HTTPException(status_code=400, detail="All data must be of type int64")

    # categorical conversion
    categorical_array = np.array(data_array[:, categorical_columns], dtype=np.int64)
    transformed_categorical = one_hot_encoder_onnx.run(None, {one_hot_encoder_input_name: categorical_array})
    transformed_categorical = np.array(transformed_categorical[0]).reshape(transformed_categorical[0].shape[0], -1)
    # sandard conversion
    continuous_array = np.array(data_array[:, continuous_columns], dtype=np.int64)
    transformed_continuous = standard_scaler_onnx.run(None, {standard_scaler_input_name: continuous_array})
    transformed_continuous = np.array(transformed_continuous[0]).reshape(transformed_continuous[0].shape[0], -1)
    # finalizing conversions
    data_array = np.concatenate([data_array, transformed_categorical], axis=1)
    data_array = np.delete(data_array, categorical_columns, 1)
    data_array = np.concatenate([data_array, transformed_continuous], axis=1)
    data_array = np.delete(data_array, continuous_columns, 1)

    return {"status": "success", "data_shape": data_array.shape}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
