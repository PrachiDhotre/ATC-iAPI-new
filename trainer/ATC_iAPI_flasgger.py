
import pandas as pd
import numpy as np
import joblib
import flasgger
from flasgger import Swagger
import flask
from flask import Flask, request
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

input_files_path = "/trainer/"
output_files_path = "/trainer/"
TFIDF_VECTORIZER_FILE_PATH_Impact_iAPI = 'tfidf_impact_iAPI.pkl'
TFIDF_VECTORIZER_FILE_PATH_Urgency_iAPI = 'tfidf_urgency_iAPI.pkl'
IMPACT_MODEL_iAPI_PATH = 'Impact_model_iAPI.pkl'
URGENCY_MODEL_iAPI_PATH = 'Urgency_model_iAPI.pkl'
iAPI_new_logs_file_name = 'iAPI_Experience Layer_19thFeb_to_25thFeb_Prod_Error_Warn_Exception.csv'              
existing_patterns_impact_urgency_labels_file_name = "iAPI_Patterns_Impact_Urgency_Pritority_labels.csv"  
Impact_Urgency_Priority_Mapping = "Impact_Urgency_Priority_mapping.csv"

model_impact = joblib.load(output_files_path+IMPACT_MODEL_iAPI_PATH)
#tfidf_impact = pickle.load(open(TFIDF_VECTORIZER_FILE_PATH_Impact_CatServ, 'rb'))
tfidf_impact = joblib.load(output_files_path+TFIDF_VECTORIZER_FILE_PATH_Impact_iAPI)  

model_urgency = joblib.load(output_files_path+URGENCY_MODEL_iAPI_PATH)
#tfidf_urgency = pickle.load(open(TFIDF_VECTORIZER_FILE_PATH_Urgency_CatServ, 'rb'))
tfidf_urgency = joblib.load(output_files_path+TFIDF_VECTORIZER_FILE_PATH_Urgency_iAPI) 

@app.route('/')
def welcome():
	return "Welcome All"

@app.route('/predict_impact',methods=['GET'])
def predict_impact():

    """ Let predict the Impact for the new pattern.
    ---
    parameters:        
      - name: New_Pattern
        in: query
        type: string
        required: true
     
    responses:
        200:
            description: The output impact     
    """
    pattern=request.args.get('New_Pattern')
    #pattern=[pattern]
    new_patterns_list = [pattern]
    new_patterns_list_tfidf=tfidf_impact.transform(new_patterns_list)
    new_pattern_tfidf=pd.DataFrame(new_patterns_list_tfidf.toarray(), columns=tfidf_impact.get_feature_names())        
    prediction = model_impact.predict(new_pattern_tfidf)
    impact_list = [1,2,3]
    numbers_list = [1,2,0]
    df_impact_mapping = pd.DataFrame({'Impact_label' : impact_list ,'Impact_number' :numbers_list })
    pred_impact = df_impact_mapping.loc[df_impact_mapping['Impact_number']==np.argmax(prediction),'Impact_label'].values[0]

    return "The predicted Impact for new pattern is : "+str(pred_impact)

@app.route('/predict_impact_for_multiple_patterns',methods=["POST"])
def predict_impact_for_multiple_patterns():
    """ Let predict the Impact for the new pattern.
    ---
    parameters:        
      - name: iAPI_new_logs_with_new_patterns
        in: formData
        type: file
        required: true
     
    responses:
        200:
            description: The output impact     
    """
    df_logs_with_no_existing_pattern=pd.read_csv(request.files.get("iAPI_new_logs_with_new_patterns"))
    new_patterns_list = df_logs_with_no_existing_pattern['PATTERN']
    new_patterns_list_tfidf=tfidf_impact.transform(new_patterns_list)
    new_pattern_tfidf=pd.DataFrame(new_patterns_list_tfidf.toarray(), columns=tfidf_impact.get_feature_names())        
    predictions = model_impact.predict(new_pattern_tfidf)
    impact_list = [1,2,3]
    numbers_list = [1,2,0]
    df_impact_mapping = pd.DataFrame({'Impact_label' : impact_list ,'Impact_number' : numbers_list })
    Impact_for_new_patterns = []
    for i in range(len(new_patterns_list)):
        pred_impact = df_impact_mapping.loc[df_impact_mapping['Impact_number']==np.argmax(predictions[i]),'Impact_label'].values[0]
        Impact_for_new_patterns.append(pred_impact)

    return "The predicted Impact for new pattern is : "+str(Impact_for_new_patterns)


if __name__=='__main__':
	app.run()