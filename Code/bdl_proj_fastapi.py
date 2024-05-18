from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import pandas as pd
import pickle
from tempfile import NamedTemporaryFile
from prometheus_fastapi_instrumentator import Instrumentator, metrics


app = FastAPI()

def please_preprocess(my_table): # preprocesses file contents to input array for model
    lii1=['Yes','No']
    lii2=['Event happened', 'No event']
    cols_with_unique_values1 = []
    cols_with_unique_values2 = []
    for col in my_table.columns:
          if my_table[col].isin(lii1).all():
                  cols_with_unique_values1.append(col)
    for col in my_table.columns:
          if my_table[col].isin(lii2).all():
                  cols_with_unique_values2.append(col)
    #pd.set_option('future.no_silent_downcasting', True)
    my_ye=my_table[cols_with_unique_values1].replace(['Yes','No'],[1,0]).astype('int64')
    my_eve=my_table[cols_with_unique_values2].replace(['Event happened','No event'],[1,0]).astype('int64')
    my_table2=my_table.copy()
    ccc=[elem for elem in  cols_with_unique_values1+cols_with_unique_values2]
    my_table_modify=my_table2.drop(ccc,axis=1)
    my_table_modify=pd.concat([my_table_modify,my_ye,my_eve],axis=1)
    my_table_str=my_table_modify.select_dtypes(exclude=['number'])
    my_table_num=my_table_modify.select_dtypes(include=['number'])
    enco = joblib.load('onehot.joblib')
    tabl=enco.transform(my_table_str)
    tabl=pd.DataFrame(tabl.toarray(),columns=enco.get_feature_names_out())
    ftable=pd.concat([tabl,my_table_num],axis=1)
    return ftable.values

def please_output(my_data): # LOads pretrained models and generates output from input array
    track_dict={}
    test_list=['EFS','DEAD','GF','AGVHD','CGVHD']
    for i in range(5):
        loaded_model = joblib.load(f'{test_list[i]}_model.joblib')
        out=loaded_model.predict_proba(my_data)[0][1]
        track_dict[test_list[i]]=out
    return track_dict

Instrumentator().instrument(app).expose(app)

@app.post('/predict')
async def predict(input_data_pkl: UploadFile = File(...)):
    try:
        # Read the contents of the uploaded file
        contents = await input_data_pkl.read()

        # Load data from the uploaded .pkl file
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(contents)
            tmp.seek(0)
            test = pickle.load(tmp)

        # Create a DataFrame from the loaded data
        testc = pd.DataFrame([test], columns=np.arange(len(test)))

        # Preprocess the data
        test_input = please_preprocess(testc)

        # Get ouptput
        out_dict = please_output(test_input)

        return {"predictions": out_dict}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)