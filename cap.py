import warnings
warnings.filterwarnings('ignore')
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
import seaborn as sns
import utils
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
import plotly.express as px
# import streamlit
import streamlit as st


# getRFMValue
def getRFMValue(pRecency,pFrequency,pMonetary):

    # read dataset
    df = pd.read_excel(".\data\Dep.xlsx")

    # # assign values
    # pRecency = 700
    # pFrequency = 200
    # pMonetary = 180000


    # create data dict ... colNames should be kay
    data = {'Recency': pRecency,
            'Frequency' : pFrequency,
            'Monetary': pMonetary}

    # create temp data frame for predict
    dft = pd.DataFrame(data, index=[0])
    #print(dft)

    # index of new record
    vIndx = df.shape[0]
    #print(vIndx)

    # concat dfs
    df = pd.concat([df, dft])
    #print(df.tail())

    # reindex
    df = df.reset_index(drop=True)
    #print(df.tail())


    # # print from dataframe
    # print('Recency  :',df['Recency'][vIndx])
    # print('Frequency:',df['Frequency'][vIndx])
    # print('Monetory :',df['Monetary'][vIndx])

    # rank
    df['R_rank'] = df['Recency'].rank(ascending=False)
    df['F_rank'] = df['Frequency'].rank(ascending=True)
    df['M_rank'] = df['Monetary'].rank(ascending=True)
     

    # normalizing the rank of the customers
    df['R_rank_norm'] =(df['R_rank']/df['R_rank'].max())*100
    df['F_rank_norm'] =(df['F_rank']/df['F_rank'].max())*100
    df['M_rank_norm'] =(df['F_rank']/df['M_rank'].max())*100    
    
    # predict

    df['RFM'] =  ((0.15 * df['R_rank_norm']) + (0.28 * df['F_rank_norm']) + (0.57 * df['M_rank_norm'])) * 0.05
    # print("RFM Score:",fRFM[vIndx])
    ##############################################
    # Customer Loyalty based upon the Frequency of the visits
    ##############################################
    #frequency >141 : Diamond
    #140 >rfm score >101 : Platinum
    # 100>frequency >51 : Gold
    # 50>frequency>21 : Silver
    # frequency <20 :Bronze

    df['Loyalty'] =np.where(df['Frequency'] > 141 , "Diamond", (np.where(df['Frequency'] >101, "Platinum", (np.where(df['Frequency'] >51,"Gold",np.where(df['Frequency'] >21 ,'Silver', 'Bronze'))))))
    

    # return (df['RFM'][vIndx])
    return (df['RFM'][vIndx],df['Loyalty'][vIndx])



# show web page & get input

from PIL import Image
img = Image.open("rfm.jpg")
#img.encode().decode('unicode_escape')
st.image(img, width=700)

st.title("RFM Score Calculator & Customer Loyalty")

st.subheader("Customer Details:")

vRecency = st.text_input('Recency') #,0,700)
if len(vRecency) > 0:
    try:
        vRecency = float(vRecency)
    except:
        st.error("Input Error")

vFrequency = st.text_input('Frequency') #,0,200)
if len(vFrequency) > 0:
    try:
        vFrequency = float(vFrequency)
    except:
        st.error("Input Error")

vMonetary = st.text_input('Monetary') #,0,180000)
if len(vMonetary) > 0:
    try:
        vMonetary = float(vMonetary)
    except:
        st.error("Input Error")


# print type
print(type(vRecency))
print(type(vFrequency))
print(type(vMonetary))


# submit
if(st.button("Submit")):

    # change type
    vRecency = float(vRecency)
    vFrequency = float(vFrequency)
    vMonetary = float(vMonetary)

    # call rfm function
    # vRFM, = getRFMValue(vRecency,vFrequency,vMonetary)
    vRFM,vloyalty = getRFMValue(vRecency,vFrequency,vMonetary)



    st.subheader('Prediction')
    st.write('========================================')
    st.write('RFM Score for the Customer : ', vRFM)

    st.write('Customer Loyalty : ', vloyalty)
    st.write('========================================')
    # reset    
    st.button("Reset")

