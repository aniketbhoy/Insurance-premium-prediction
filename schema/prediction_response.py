from pydantic import BaseModel, Field
from typing import Dict

class PredictionResponse(BaseModel):
    predicted_category: str=Field(..., description='the predicted insurance premium category',example='High')
    confidence: float=Field(...,description="model's confidence score for predicted class (range: 0 to 1)", example=0.8432)
    class_probabilities: Dict[str,float]=Field(...,description='probability description across all possible classes', example={'Low':0.01,'Medium':0.15,'High':0.84})
