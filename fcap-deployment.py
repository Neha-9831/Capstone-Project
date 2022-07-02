# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# imports
import pandas as pd
# matplotlib 
import matplotlib.pyplot as plt
# sns
import seaborn as sns
# plotly ex
import plotly.express as px
# import streamlit
import streamlit as st
import numpy as np


# load model
print("\n*** Load Model ***")
import pickle
Fcap = '.\data\Capstone.pkl'
model = pickle.load(open(Fcap, 'rb'))
print(model)
print("Done ...")


# load vars
print("\n*** Load Vars ***")
Fcap = '.\data\CapVar.pkl'
dVars = pickle.load(open(Fcap, 'rb'))
print(dVars)
clsVars = dVars['clsvars']
#clsVars = np.reshape(clsVars,[])
#clsVars1 = clsVars.reshape([])
#clsVars1 = clsVars.values.reshape()
clsVars1 = np.column_stack((clsVars))
allCols = dVars['allCols']
le = dVars['le'] 
print("Done ...")


################################
# predict
#######N#######################

def getPredict(dfp):
    global clsVars, allCols, LeSpc
   
    # split into data & outcome
    print("Step 1")
    X_pred = dfp[allCols].values
    print(X_pred.shape)
#    y_pred = dfp[clsVars].values
    # predict from model
    print("Step 2")
    p_pred = model.predict(X_pred)
    # update data frame
    print("Step 3")
    dfp['Predict'] = p_pred
    #dfp[clsVars] = le.inverse_transform(dfp[clsVars])
    #dfp['Predict'] = le.inverse_transform(dfp['Predict'])
    return (dfp)



st.title("Understanding Customer Behaviour with Customer Segmentation")


from PIL import Image
img = Image.open("final.jpg")
st.image(img, width=750)

st.subheader("Customer Segemnetation")

vrfm = st.text_input('RFM Score: ')
if len(vrfm) > 0:
    try:
        vrfm = float(vrfm)
    except:
        st.error("Input Error")

cID = st.text_input('Customer ID: ')
if len(cID) > 0:
    try:
        cID = float(cID)
    except:
        st.error("Input Error")

loyalty = st.selectbox("Customer Loyalty: ",
                         ['0', '1', '2', '3', '4']) 

if (loyalty == '0'):

    st.success("Bronze")

elif (loyalty == '1'):

    st.success("Silver")

elif (loyalty == '2'):

    st.success("Gold")

elif (loyalty == '3'):

    st.success("Platinum")

else:

    st.success("Diamond")

    

if(st.button("Submit")):

     # change type
    
    vrfm = float(vrfm)
    cID = float(cID)
    loyalty = int (loyalty)
    # create data dict ... colNames should be kay
    data = {'CustomerID': cID,
            'RFM_Score': vrfm,
            'Customer_Loyalty' : loyalty}

    # create data frame for predict
    dfp = pd.DataFrame(data, index=[0])
    # show dataframe
    st.subheader('Input Data')
    st.write('Customer ID: ', cID)
    st.write('RFM Score: ', vrfm)
    st.write('Customer Loyalty : ', loyalty)

    
    # # predict
    print("Now Predict")
    dfp = getPredict(dfp)

    # show dataframe

    st.subheader('Customer Segmentation')
    
    st.write('Customer Segmentation (0-High Value Customer, 1-Medium Value Customer, 2-Low Value Customer ) : ', dfp['Predict'][0])

    # reset    
    st.button("Reset")