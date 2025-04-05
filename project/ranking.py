import streamlit as st
import os

import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import re

def to_tag(x):
    try:
        tag1=re.findall(r"#\w+", x)  
    except:
        tag1=[None]
                
    try:            
        tag2=re.findall(r"@\w+", x)
    except:
        tag2=[None]
    tag=tag1+tag2
    return tag


def to_int(value):
    if isinstance(value, (int, float)):
        return value

    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000, 'T': 1_000_000_000_000}
    try:
        if value[-1] in multipliers:
            return int(float(value[:-1]) * multipliers[value[-1]])  

        return int(value)  
    except:
        return 0
def get_progress_color(value):
    if value >= 80:
        return "#4CAF50"  # Green (80~100)
    elif value >= 60:
        return "#2196F3"  # Blue (60~80)
    elif value >= 40:
        return "#800080"  # Purple (40~60)
    elif value >= 20:
        return "#FFA500"  # Orange (20~40)
    else:
        return "#FF0000"  # Red (0~20)
    
def calculate_rise_rate_growth(series: pd.Series, alpha: float = 0.1):
    ema = series.ewm(alpha=alpha).mean()
    growth_rate_ema = ((series - ema) / ema) * 100

    df_ema = pd.DataFrame({
            "Value": series,
            "EMA": ema,
            "Growth_Rate(%)": growth_rate_ema
        })
    return growth_rate_ema.iloc[-5:].mean()


def rank():
    st.set_page_config(
        page_title="Demo",
        page_icon="📌",
        layout="wide",
        initial_sidebar_state="expanded")

    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../loaded_influencer_data"))
    influencer_list = os.listdir(folder_path)
    ranking_data=pd.DataFrame(columns=['profile','influencer','total_ER','total_view',
                                    "Fixed_video_ER","Fixed_like_ER","Fixed_comment_ER",
                                    "Fixed_saved_ER","Non-fixed_video_ER","Non-fixed_like_ER",
                                    "Non-fixed_comment_ER","Non-fixed_saved_ER"])
    tag_fig={}

    for name in influencer_list:
        tmp=pd.DataFrame(index=[0],columns=['profile','influencer','total_ER','total_view',
                                            "Fixed_video_ER","Fixed_like_ER","Fixed_comment_ER",
                                            "Fixed_saved_ER","Non-fixed_video_ER","Non-fixed_like_ER",
                                            "Non-fixed_comment_ER","Non-fixed_saved_ER"])

        try:

            profile_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../loaded_influencer_data/"+name+"/"+name+".xlsx"))
            video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../loaded_influencer_data/"+name+"/"+name+"_video.xlsx"))

            profile=pd.read_excel(profile_path, usecols = "A,B,C,D")
            video=pd.read_excel(video_path, usecols = "A,B,C,D,E,F,G,H,I,J,K,L,M")         
        except:
            continue 
             
        video['tag']=video['content'].apply(lambda x : to_tag(x))

        for i in range(len(video)):
            for l in video.loc[i,'tag']:
                if l in list(tag_fig.keys()):
                    if name in list(tag_fig[l].keys()):
                        tag_fig[l][name]+=1    
                    else:
                        tag_fig[l][name]=int(1)
                        
                else :
                    tag_fig[l]={name:int(1)}


        
        tmp['profile']= folder_path+'/'+name+"/profile.jpg"#[np.array([img])]
        tmp['influencer']=name
        
        tmp['total_ER']=profile.iloc[-1,1]/profile.iloc[-1,2]
        tmp['total_view']=video['views'].sum()
        
        tmp['total_ER_rise_rate']=calculate_rise_rate_growth(profile['total_likes']/profile['followers'])#.rolling(window=3).mean().dropna()]
        tmp['total_view_rise_rate']=calculate_rise_rate_growth(video['views'])#.rolling(window=3).mean().dropna()]
            
        tmp['Fixed_video_ER']=(video.loc[video['badge']=='고정됨','video_ER']).mean()
        tmp['Fixed_like_ER']=(video.loc[video['badge']=='고정됨','like_ER']).mean()
        tmp['Fixed_comment_ER']=(video.loc[video['badge']=='고정됨','comments_ER']).mean()
        tmp['Fixed_saved_ER']=(video.loc[video['badge']=='고정됨','saved']/video.loc[video['badge']=='고정됨','views']).mean()

        tmp['Fixed_video_ER_rise_rate']=calculate_rise_rate_growth(video.loc[video['badge']=='고정됨','video_ER'])#.rolling(window=3).mean().dropna()]
        tmp['Fixed_like_ER_rise_rate']=calculate_rise_rate_growth(video.loc[video['badge']=='고정됨','like_ER'])#.rolling(window=3).mean().dropna()]
        tmp['Fixed_comment_ER_rise_rate']=calculate_rise_rate_growth(video.loc[video['badge']=='고정됨','comments_ER'])#.rolling(window=3).mean().dropna()]
        tmp['Fixed_saved_ER_rise_rate']=calculate_rise_rate_growth(video.loc[video['badge']=='고정됨','saved']/video.loc[video['badge']=='고정됨','views'])#.rolling(window=3).mean().dropna()]

        tmp['Non-fixed_video_ER']=(video.loc[video['badge']!='고정됨','video_ER']).mean()
        tmp['Non-fixed_like_ER']=(video.loc[video['badge']!='고정됨','like_ER']).mean()
        tmp['Non-fixed_comment_ER']=(video.loc[video['badge']!='고정됨','comments_ER']).mean()
        tmp['Non-fixed_saved_ER']=(video.loc[video['badge']!='고정됨','saved']/video.loc[video['badge']!='고정됨','views']).mean()

        tmp['Non-fixed_video_ER_rise_rate']=calculate_rise_rate_growth(video.loc[video['badge']!='고정됨','video_ER'])#.rolling(window=3).mean().dropna()]
        tmp['Non-fixed_like_ER_rise_rate']=calculate_rise_rate_growth(video.loc[video['badge']!='고정됨','like_ER'])#.rolling(window=3).mean().dropna()]
        tmp['Non-fixed_comment_ER_rise_rate']=calculate_rise_rate_growth(video.loc[video['badge']!='고정됨','comments_ER'])#.rolling(window=3).mean().dropna()]
        tmp['Non-fixed_saved_ER_rise_rate']=calculate_rise_rate_growth(video.loc[video['badge']!='고정됨','saved']/video.loc[video['badge']!='고정됨','views'])#.rolling(window=3).mean().dropna()]


        ranking_data=pd.concat([ranking_data,tmp])
        
    df = pd.DataFrame(ranking_data).fillna(0)

    with st.sidebar:
        st.title('📌 other menu ')
        

        if st.button('EXIT'):
            st.session_state['authenticated'] = False        
            st.rerun()

        if st.button('main'):
            st.session_state['authenticated'] = 'main'
            st.rerun()

        if st.button('tracking'):
            st.session_state['authenticated'] = 'tracking'        
            st.rerun()

        
        
        st.title("📊 Ranking")
        
        type_options =st.sidebar.radio(
                "Metric Type",["Mean","Rise Rate",'Tag'],horizontal=False)


        
        ranking_options_default = ['total_ER','total_view',
                                    "Fixed_video_ER","Fixed_like_ER","Fixed_comment_ER",
                                    "Fixed_saved_ER","Non-fixed_video_ER","Non-fixed_like_ER",
                                    "Non-fixed_comment_ER","Non-fixed_saved_ER"]

        ranking_options_rise_rate = ['total_ER_rise_rate','total_view_rise_rate',
                                    "Fixed_video_ER_rise_rate","Fixed_like_ER_rise_rate","Fixed_comment_ER_rise_rate",
                                    "Fixed_saved_ER_rise_rate","Non-fixed_video_ER_rise_rate","Non-fixed_like_ER_rise_rate",
                                    "Non-fixed_comment_ER_rise_rate","Non-fixed_saved_ER_rise_rate"]

        if type_options=="Mean":
            ranking_options=ranking_options_default

            selected_metric = st.selectbox("Select Ranking Metric", ranking_options, index=0)        
            
            st.markdown("Select Sub Metric")
            selected_options = []
            for option in ranking_options:
                if st.checkbox(option):
                    selected_options.append(option)

        elif type_options=="Rise Rate":
            ranking_options=ranking_options_rise_rate

            selected_metric = st.selectbox("Select Ranking Metric", ranking_options, index=0)        
            
            st.markdown("Select Sub Metric")
            selected_options = []
            for option in ranking_options:
                if st.checkbox(option):
                    selected_options.append(option)

        else : 
            selected_tag=st.selectbox("Select Ranking Tag", tag_fig.keys(), index=0)        

            
        
    if type_options=="Tag":
    
        cols = st.columns((3, 3, 3, 3, 3, 3), gap='large',border=False,vertical_alignment="top")  
        
        for i, row in enumerate(sorted(tag_fig[selected_tag].items(), key=lambda x: x[1],reverse=True)):

            profile=os.path.abspath(os.path.join(os.path.dirname(__file__), "../loaded_influencer_data/"+row[0]+"/profile.jpg"))
            influencer=row[0]
            if i==0:
                rank='🥇 1 st '
            elif i==1:
                rank='🥈 2 nd '
            elif i==2:
                rank='🥉 3 rd '
            else :
                rank=str(i+1)+'th '

            with cols[i % 6]:  
                st.markdown(f'<p style=font-size: 10px; font-weight: bolder;">{influencer}</p>', unsafe_allow_html=True)
                st.image(profile,use_container_width=True) 
                        
                content=rank#+str(row[1])
                st.markdown(f'<p style=font-size: 3px;">{content}</p>', unsafe_allow_html=True)
                
    
    else:
        df_sorted = df.sort_values(by=selected_metric, ascending=False).reset_index(drop=True)

        top_n = 30
        top_ranked = df_sorted.head(top_n)

        scaler = MinMaxScaler()
        
        top_ranked[ranking_options] = scaler.fit_transform(top_ranked[ranking_options])*100

        cols = st.columns((3, 3, 3, 3, 3, 3), gap='large',border=False,vertical_alignment="top") 
        for i, row in enumerate(top_ranked.itertuples(index=False, name=None)):
            profile, influencer, *metrics = row  
            normalized_metrics = top_ranked.iloc[i]
            if i==0:
                rank='🥇 1 st '
            elif i==1:
                rank='🥈 2 nd '
            elif i==2:
                rank='🥉 3 rd '
            else :
                rank=str(i+1)+'th '

            with cols[i % 6]:  
                #st.markdown(f"# {influencer}")
                st.markdown(f'<p style=font-size: 10px; font-weight: bolder;">{influencer}</p>', unsafe_allow_html=True)

                st.image(profile,use_container_width=True) 
                
                content=rank+selected_metric
                st.markdown(f'<p style=font-size: 3px;">{content}</p>', unsafe_allow_html=True)
                value = int(normalized_metrics[selected_metric])
                color = get_progress_color(value)
                st.markdown(
                            f'<div style="width: 100%; height: 10px; background: lightgray; border-radius: 5px;">'
                            f'<div style="width: {value}%; height: 100%; background: {color}; border-radius: 5px;"></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                st.markdown("---")

                
                
                for metric in selected_options:
                    if metric != selected_metric :
                        
                        content=metric#+" : "+ str(np.around(normalized_metrics[metric],2))+"%"
                        st.markdown(f'<p style=font-size: 0.5px;">{content}</p>', unsafe_allow_html=True)
                        value = int(normalized_metrics[metric])
                        color = get_progress_color(value)
                        st.markdown(
                            f'<div style="width: 100%; height: 10px; background: lightgray; border-radius: 5px;">'
                            f'<div style="width: {value}%; height: 100%; background: {color}; border-radius: 5px;"></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        #st.progress(value / 100) 
                        
                #st.markdown("---")
            
            