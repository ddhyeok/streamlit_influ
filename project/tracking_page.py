import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
import re
import streamlit as st
import yaml
import random
import plotly.subplots as sp

def tracking():
    with st.sidebar:
        st.title('📌 other menu ')
        

        if st.button('EXIT'):
            st.session_state['authenticated'] = False        
            st.rerun()

        if st.button('main'):
            st.session_state['authenticated'] = 'main'
            st.rerun()

        if st.button('ranking'):
            st.session_state['authenticated'] = 'ranking'        
            st.rerun()

    st.title(st.session_state['user']+"'s Tracking URL")
    df_path = os.path.join(os.path.dirname(__file__), "../user/"+st.session_state['user']+"/url_list.xlsx")

    df=pd.read_excel(df_path)        
    #st.dataframe(df)
    st.dataframe(
        df,
        column_config={
            "Description": "Description",
            "url": st.column_config.LinkColumn("URL"),
            "likes": st.column_config.LineChartColumn(
                "likes", 
            ),
            "comments": st.column_config.LineChartColumn(
                "comments", 
            ),
            "shared": st.column_config.LineChartColumn(
                "shared", 
            ),
            "saved": st.column_config.LineChartColumn(
                "saved",
            ),
            "time": "Last updated time",


        },
        #hide_index=True,
    )
    

    TIKTOK_URL_PATTERN = r"^https?://(www\.)?tiktok\.com/@[\w.-]+/video/\d+/?$"

    def is_valid_tiktok_url(url: str) -> bool:
        return re.match(TIKTOK_URL_PATTERN, url) is not None

    Description = st.text_input("🔗 Insert Description", placeholder="Insert Description...")    
    insert_url = st.text_input("🔗 Insert URL", placeholder="Insert URL...")

    if st.button("🔗 Insert URL"):
        if insert_url in df["url"].values:
            st.warning("⚠️ Duplicate URL !!")
        elif Description in df["Description"].values:
            st.warning("⚠️ Duplicate Description !!")

        elif not is_valid_tiktok_url(insert_url):
            st.warning("⚠️ No Tiktok url !!")


        elif insert_url == '':
            st.warning("⚠️ No Url !!")

        elif Description =="":
            st.warning("⚠️ No Description !!")

        else:
            
            new= {"Description":[Description],"url": [insert_url],"likes":[[]],"comments":[[]],"shared":[[]],"saved":[[]]}
            df = pd.concat([df, pd.DataFrame({
    "Description":[Description],
    "url": [insert_url],
    "likes": [[]],  #
    "comments": [[]],
    "shared": [[]],
    "saved": [[]]
}, dtype=object)])
            

            df.to_excel(df_path, index=False)
            st.rerun()
            

    delete_Description = st.selectbox("🗑 Delete URL, input Description", df["Description"].unique() if not df.empty else [""], index=None, placeholder="Select URL...")
    if st.button("🗑 Delete URL "):
        if delete_Description in df["Description"].values:
            df = df[df["Description"] != delete_Description]


            df.to_excel(df_path, index=False)
            st.rerun()
        else:
            st.warning("⚠️ No Description !!")




    
    st.markdown("---")
    if len(df)>=1 :       
        
        selected = st.selectbox("Select Your URL", df["Description"])

        entity_data = df[df["Description"] == selected].iloc[0]

        ts_df = pd.DataFrame({
        "likes": eval(entity_data["likes"]),
        "comments": eval(entity_data["comments"]),
        "shared": eval(entity_data["shared"]),
        "saved": eval(entity_data["saved"]),
    })

        fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("Likes", "Comments", "Shared", "Saved"))

        fig.add_trace(go.Scatter( y=ts_df["likes"], mode="lines+markers", name="likes"), row=1, col=1)
        fig.add_trace(go.Scatter( y=ts_df["comments"], mode="lines+markers", name="comments"), row=1, col=2)
        fig.add_trace(go.Scatter( y=ts_df["shared"], mode="lines+markers", name="shared"), row=2, col=1)
        fig.add_trace(go.Scatter( y=ts_df["saved"], mode="lines+markers", name="saved"), row=2, col=2)

        fig.update_layout(height=600, width=1000, title_text="📈 "+selected+" details", showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

        