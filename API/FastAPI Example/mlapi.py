# Import dependencies
from fastapi import FastAPI
from pydantic import BaseModel # pass json information to the api
import pickle
import pandas as pd

app = FastAPI() # instance to the class

# structure in which the API is goign to receive the request
class ScoringItem(BaseModel):
    YearsAtCompany:float #1
    EmployeeSatisfaction: float # 0.01
    Position: str # Manager or Non-Manager
    Salary : int # Ordinal 1,2,3,4,5

with open('rfmodel.pkl', 'rb') as f: # rb sets file as binary
    model = pickle.load(f) 
# decorator to tell the API what the route is going to be
@app.get('/')
async def scoring_endpoint(): # this is get called once we go to above route
    return {"hello":"world"}

@app.post('/app')
async def scoring_endpoint(item:ScoringItem): # this is get called once we go to above route
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction": int(yhat)} 