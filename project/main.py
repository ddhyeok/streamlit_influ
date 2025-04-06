import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import plotly.express as px
import os 
import tool as t
import yaml



def main_page():
    st.set_page_config(
        page_title="Demo",
        page_icon="📌",
        layout="wide",
        initial_sidebar_state="expanded")
    

    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../loaded_influencer_data"))
    influencer_list = os.listdir(folder_path)



    with st.sidebar:

        if st.button('EXIT'):
            st.session_state['authenticated'] = False        
            st.rerun()

        if st.button('ranking'):
            st.session_state['authenticated'] = 'ranking'        
            st.rerun()

        if st.button('tracking'):
            st.session_state['authenticated'] = 'tracking'        
            st.rerun()

        st.title('📌 Select your Influencer')
        
        selected_influencer = st.selectbox('Select your Influencer', influencer_list, index=len(influencer_list)-1)

    try:
            loaded=t.dataset(selected_influencer)
            
            loaded.prepro()
            loaded.making_tag()
            
            video_pie,like_pie,shared_pie,comments_pie,count_pie = loaded.making_pie_g()
            
            line1,line2,line3,line4,line5=loaded.fig1()
            bar=loaded.fig2()
            heatmap= loaded.making_heatmap()
            
            profile_fig=loaded.profile_fig()

            Col = st.columns((1.5,4), gap='medium')
            with Col[0]:
                
                st.markdown('#### '+selected_influencer)
                st.image(loaded.img, caption= "https://www.tiktok.com/@"+selected_influencer, use_container_width=True)

            with Col[1]:

                col = st.columns((1,1,1,1), gap='medium')


                with col[0]:
                    st.markdown('#### Gains/Losses')
                    try:
                        st.metric(label="likes",value=t.to_str(loaded.profile['total_likes'].tolist()[-1]),
                            delta=t.to_str(loaded.profile['total_likes'].tolist()[-1]-loaded.profile['total_likes'].tolist()[-7]))
                    except:
                        st.metric(label="likes",value=t.to_str(loaded.profile['total_likes'].tolist()[-1]))
                with col[1]:
                    st.markdown('#### ')
                    try:
                        st.metric(label="Followers",value=t.to_str(loaded.profile['followers'].tolist()[-1]),
                            delta=t.to_str(loaded.profile['followers'].tolist()[-1]-loaded.profile['followers'].tolist()[-7]))
                    except:
                        st.metric(label="Followers",value=t.to_str(loaded.profile['followers'].tolist()[-1]))
                
                with col[2]:
                    st.markdown('#### ')
                    try:
                        st.metric(label="Following",value=t.to_str(loaded.profile['following'].tolist()[-1]),
                            delta=t.to_str(loaded.profile['following'].tolist()[-1]-loaded.profile['following'].tolist()[-7]))
                    except:
                        st.metric(label="Following",value=t.to_str(loaded.profile['following'].tolist()[-1]))
                
                with col[3]:
                    st.markdown('#### ')
                    try:
                        st.metric(label="Total ER",value=t.to_str(np.round(loaded.profile['ER'].tolist()[-1],2))+'%',
                            delta=t.to_str(np.round(loaded.profile['ER'].tolist()[-1]-loaded.profile['ER'].tolist()[-7],3)))
                    except:
                        st.metric(label="Total ER",value=t.to_str(np.round(loaded.profile['ER'].tolist()[-1],2))+'%')

                st.plotly_chart(profile_fig, use_container_width=True)    
                
                

            st.subheader("📈 Recent Video Engagement Rate")
            #st.plotly_chart(line1, use_container_width=True)    
            #st.plotly_chart(line2, use_container_width=True)    
            st.plotly_chart(line3, use_container_width=True)    
            #st.plotly_chart(line4, use_container_width=True)    
            st.plotly_chart(line5, use_container_width=True)    


            c= st.columns((1, 1, 1, 1))  
            with c[0]:
                st.plotly_chart(video_pie, use_container_width=True)
            with c[1]:
                st.plotly_chart(comments_pie, use_container_width=True)
            with c[2]:
                st.plotly_chart(like_pie, use_container_width=True)
            with c[3]:
                st.plotly_chart(count_pie, use_container_width=True)
                
            #    st.plotly_chart(shared_pie, use_container_width=True)
                

            cc= st.columns((1, 1))  
            with cc[0]:
                st.subheader("📊 Average Engagement Rate")
                st.plotly_chart(bar, use_container_width=True)
            with cc[1]:
                st.subheader("🔥 Hashtag Combination Engagement Rate (Heatmap)")
                st.plotly_chart(heatmap, use_container_width=True)
            #with cc[2]:
            #    st.plotly_chart(count_pie, use_container_width=True)

    except:
            st.text("No Data")
