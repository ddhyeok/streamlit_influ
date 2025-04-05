from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt

import os
import re
import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class dataset():
    def __init__(self,name):
        

        self.profile_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../loaded_influencer_data/"+name+"/"+name+".xlsx"))
        self.video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../loaded_influencer_data/"+name+"/"+name+"_video.xlsx"))
        self.img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../loaded_influencer_data/"+name+"/profile.jpg"))
        

        self.profile=pd.read_excel(self.profile_path, usecols = "A,B,C,D")
        self.video=pd.read_excel(self.video_path, usecols = "A,B,C,D,E,F,G,H,I,J,K,L,M,N")      
        self.img=Image.open(self.img_path)
        
        
        self.profile['total_likes']=self.profile['total_likes'].apply(lambda x: to_int(x))
        self.profile['followers']=self.profile['followers'].apply(lambda x: to_int(x))
        self.profile['following']=self.profile['following'].apply(lambda x: to_int(x))

        
        pass
    def prepro(self):
        
        
        '''
        
        self.profile["timestamp"] = pd.to_datetime(self.profile["timestamp"])

        self.profile["date"] = self.profile["timestamp"].dt.date

        self.profile = self.profile.loc[
            self.profile.groupby("date")[["total_likes", "followers", "following"]].idxmax().max(axis=1)
        ]

        self.profile = self.profile.sort_values("date")
        self.profile=self.profile.drop(columns=['timestamp']).reset_index(drop=True)
        '''
        self.profile=self.profile.rename(columns={'timestamp':'date'})
        self.profile['ER']=self.profile['total_likes']/self.profile['followers']

        self.video['tag']=self.video['content'].apply(lambda x : to_tag(x))
        self.video['saved_ER']=self.video['saved']/self.video['views']
        self.video['comments_ER']=self.video['comments']/self.video['views']

        #self.video['date']=self.video['date'].apply(lambda x: dt.strptime(x ,"%Y-%m-%d"))

        self.video["date"] = pd.to_datetime(self.video["date"])
        self.video['date'] = self.video['date'].interpolate(method='linear')
        self.video['date'] = self.video['date'].fillna(method='bfill')  
        self.video['date'] = self.video['date'].fillna(method='ffill') 

        self.video['badge']=self.video['badge'].apply(lambda x : "non-fixed" if x==0 else "fixed")


    def profile_fig(self):
        
        
        df=self.profile.copy()
        scaler = MinMaxScaler()

        df[["total_likes", "followers", "following", "ER"]] = scaler.fit_transform(df[["total_likes", "followers", "following", "ER"]])

        df = df.melt(id_vars=["date"], var_name="metric", value_name="value")

        fig = px.line(
                    df,
                    x="date", 
                    y="value", 
                    color="metric",  
                    facet_row="metric",  
                    title="Metric Trends Over Time",
                    height=450) 

        fig.update_yaxes(title_text="", showticklabels=False)  
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("metric=", "").title()))

        return fig

    def fig1(self): 
        '''
        df_melted = pd.melt(self.video, 
                            id_vars=["views", "likes", "comments", "shared", "saved", "content", "badge","date"],
                            value_vars=["video_ER", "like_ER", "comments_ER"],#, "shared_ER"], 
                            var_name="metric",
                            value_name="ER_value")
        filtered_df = df_melted[df_melted['badge'] == 'non-fixed']


        fig1 = px.line(
            filtered_df,
            x="date",
            y="ER_value",
            color="metric",
            line_shape="linear",
            height=500,
            width=900,
            title="Metric Trends Over Time(original ER Value)",            
        )
        for trace in fig1.data:
            trace.showlegend = False

        scatter_fig1 = px.scatter(
            filtered_df,
            x="date",
            y="ER_value",
            color="metric",
            trendline="ols", 
            opacity=0.6,
        )

        fig1.add_traces(scatter_fig1.data)  
#########################
        df_melted=pd.melt(self.video, 
                            id_vars=[ "content", "badge","date"],
                            value_vars=["views", "likes", "comments", "saved"],#,"shared"],#["video_ER", "like_ER", "shared_ER", "comments_ER"], 
                            var_name="metric",
                            value_name="value")
        filtered_df = df_melted[df_melted['badge'] == 'non-fixed']

        fig2 = px.line(
            filtered_df,
            x="date",
            y="value",
            color="metric",
            line_shape="linear",
            height=500,
            width=900,
            title="Metric Trends Over Time(original Value)",
            )
        for trace in fig2.data:
            trace.showlegend = False

        scatter_fig2 = px.scatter(
            filtered_df,
            x="date",
            y="value",
            color="metric",
            trendline="ols", 
            opacity=0.6,
        )

        fig2.add_traces(scatter_fig2.data)
###############
    '''
        df=self.video.copy()
        scaler = MinMaxScaler()
        df[["views", "likes", "shared", "saved","comments",
            "video_ER", "like_ER", "shared_ER", "comments_ER","saved_ER"
            ]] = scaler.fit_transform(df[["views", "likes", "shared", "saved","comments",
                                          "video_ER", "like_ER", "shared_ER", "comments_ER",
                                          "saved_ER"]])

        df_melted = pd.melt(df, 
                            id_vars=[ "badge","date"],
                            value_vars=["video_ER", "like_ER",  "comments_ER","saved_ER"],#"shared_ER"], 
                            var_name="metric",
                            value_name="ER_value")
        filtered_df = df_melted[df_melted['badge'] == 'non-fixed']


        fig3 = px.line(
            filtered_df,
            x="date",
            y="ER_value",
            color="metric",
            line_shape="linear",
            height=500,
            width=900,
            title="Metric Trends Over Time(MMscaled original ER Value)",
        )
        for trace in fig3.data:
            trace.showlegend = False

        scatter_fig3= px.scatter(
            filtered_df,
            x="date",
            y="ER_value",
            color="metric",
            trendline="ols", 
            opacity=0.6,
        )

        fig3.add_traces(scatter_fig3.data)
        ##################################
        df_melted_ori=pd.melt(df, 
                            id_vars=[ "badge","date"],
                            value_vars=["views", "likes", "comments", "saved"],#,"shared"],#["video_ER", "like_ER", "shared_ER", "comments_ER"], 
                            var_name="metric",
                            value_name="value")
        filtered_df_ori = df_melted_ori[df_melted_ori['badge'] == 'non-fixed']
        fig5 = px.line(
                    filtered_df_ori,
                    x="date", 
                    y="value", 
                    color="metric",  
                    facet_row="metric",  
                    title="Metric Trends Over Time(MMscaled original Value)",
                    height=450) 
        
        '''
        
        fig4 = px.line(
            filtered_df,
            x="date",
            y="value",
            color="metric",
            line_shape="linear",
            height=500,
            width=900,
            title="Metric Trends Over Time(MMscaled original Value)",
        )
        for trace in fig4.data:
            trace.showlegend = False

        scatter_fig4 = px.scatter(
            filtered_df,
            x="date",
            y="value",
            color="metric",
            trendline="ols", 
            opacity=0.6,
        )

        fig4.add_traces(scatter_fig4.data)
        '''
        ############

        fig5.update_yaxes(title_text="",showticklabels=False)  
        fig5.for_each_annotation(lambda a: a.update(text=a.text.replace("metric=", "").title()))
    
        return '','',fig3,'', fig5
        #return fig1,fig2,fig3,fig4, fig5
    
    def fig2(self): 
        df_melted = pd.melt(self.video, 
                            id_vars=["views", "likes", "comments", "shared", "saved", "content", "badge","date"],
                            value_vars=["video_ER", "like_ER", "comments_ER","saved_ER"],#, "shared_ER"], 
                            var_name="metric",
                            value_name="ER_value")
        df_grouped = df_melted.groupby(["metric", "badge"])["ER_value"].agg(
            mean="mean",
        ).reset_index()

        fig = make_subplots(
            rows=1, cols=4, 
            subplot_titles=["Video ER", "Like ER", "Comments ER", "Saved ER"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]  # 개별 Y축
        )

        metrics = ["video_ER", "like_ER", "comments_ER", "saved_ER"]
        legend_added = set() 

        for idx, metric in enumerate(metrics):
            df_metric = df_grouped[df_grouped["metric"] == metric]  
            
            for badge in df_metric["badge"].unique():
                df_badge = df_metric[df_metric["badge"] == badge]
                
                show_legend = badge not in legend_added
                label=''
                if show_legend:
                    label='%'
                    legend_added.add(badge) 

                fig.add_trace(
                    go.Bar(
                        x=[badge]*len(df_badge), 
                        y=df_badge["mean"], 
                        name=badge, 
                        showlegend=show_legend  
                    ),
                    row=1, col=idx+1
                )

            fig.update_yaxes(title_text=label, row=1, col=idx+1)

            fig.update_layout(
            title_text="Engagement Rate by Metric (Fixed vs Non-Fixed)",
            height=500, width=1200,
            showlegend=False
            )
            '''
            fig = px.bar(
            df_grouped,
            x="metric", 
            y="mean", 
            color="badge",  
            barmode="group",  
            title="Average Engagement Rate by Metric<br>(Fixed vs Non-Fixed)"
        )

        fig.update_layout(
            xaxis_title="Engagement Type",
            yaxis_title="Average Engagement Rate (%)",
            legend_title="Type",  
            height=500, width=600  
        )
        '''

        '''
        fig=plt.figure(figsize=(5, 5))
        gg=sns.set_theme(style="whitegrid")
        gg=sns.barplot(data=self.df_melted, x="metric", y="ER_value", hue="badge", palette="pastel")

        plt.xlabel("Engagement Type")
        plt.ylabel("Average Engagement Rate (%)")
        plt.title("Average Engagement Rate by Metric\n (Fixed vs Non-Fixed)")
        plt.legend(title="Type")
        '''        
        return fig
    
    def making_tag(self):
        self.tag_fig={}
        self.combi_tag={}
        
        for i in range(len(self.video)):
            for l in itertools.permutations(self.video.loc[i,'tag'], 2):
                if l in list(self.combi_tag.keys()):
                    self.combi_tag[l]['video_ER'].append(self.video.iloc[i,7])
                    self.combi_tag[l]['like_ER'].append(self.video.iloc[i,8])
                    self.combi_tag[l]['shared_ER'].append(self.video.iloc[i,9])
                    self.combi_tag[l]['comments_ER'].append(self.video.iloc[i,10])
                    
                else :
                    self.combi_tag[l]={
                        "video_ER":[self.video.iloc[i,7]],
                        "like_ER":[self.video.iloc[i,8]],
                        "shared_ER":[self.video.iloc[i,9]],
                        "comments_ER":[self.video.iloc[i,10]],
                    }
                
            for l in self.video.loc[i,'tag']:
                if l in list(self.tag_fig.keys()):
                    self.tag_fig[l]['video_ER'].append(self.video.iloc[i,7])
                    self.tag_fig[l]['like_ER'].append(self.video.iloc[i,8])
                    self.tag_fig[l]['shared_ER'].append(self.video.iloc[i,9])
                    self.tag_fig[l]['comments_ER'].append(self.video.iloc[i,10])
                    self.tag_fig[l]['Count']+=1
                    
                else :
                    self.tag_fig[l]={
                        "video_ER":[self.video.iloc[i,7]],
                        "like_ER":[self.video.iloc[i,8]],
                        "shared_ER":[self.video.iloc[i,9]],
                        "comments_ER":[self.video.iloc[i,10]],
                        "Count":1
                    }
                    
    def making_pie_g(self):
        metric=list(self.tag_fig[list(self.tag_fig.keys())[0]].keys())
        values={"video_ER":[], "like_ER":[], "shared_ER":[], "comments_ER":[],"Count":[]}
        for l in list(self.tag_fig.keys()):
            values[metric[0]].append(np.mean(self.tag_fig[l][metric[0]]))
            values[metric[1]].append(np.mean(self.tag_fig[l][metric[1]]))
            values[metric[2]].append(np.mean(self.tag_fig[l][metric[2]]))
            values[metric[3]].append(np.mean(self.tag_fig[l][metric[3]]))
            values[metric[4]].append(self.tag_fig[l][metric[4]])
        
        return create_pie_chart(list(self.tag_fig.keys()), values[metric[0]], metric[0]),\
               create_pie_chart(list(self.tag_fig.keys()), values[metric[1]], metric[1]),\
               create_pie_chart(list(self.tag_fig.keys()), values[metric[2]], metric[2]),\
               create_pie_chart(list(self.tag_fig.keys()), values[metric[3]], metric[3]),\
               create_pie_chart(list(self.tag_fig.keys()), values[metric[4]], metric[4])

    def making_heatmap(self):
        self.tag_combi_ER=pd.DataFrame(columns=['tag1','tag2','video_ER'])
        index=0
        for i in self.combi_tag.keys():
            self.tag_combi_ER.loc[index,'tag1']=i[0]
            self.tag_combi_ER.loc[index,'tag2']=i[1]
            self.tag_combi_ER.loc[index,'video_ER']=np.sum(self.combi_tag[i]['video_ER'])
            index+=1
        heatmap=px.imshow(self.tag_combi_ER.sort_values('video_ER',ascending=False
                         ).reset_index(drop=True).iloc[:70,:].pivot(
                        index='tag1',columns='tag2',values='video_ER').fillna(0),
                    text_auto=True,  
                    width=800, height=500, 
                    title="Hashtag Combination Engagement Rate (Heatmap)", 
                    labels={"video_ER": "Sum Engagement Rate (%)"},
                    color_continuous_scale='RdBu_r', origin='lower'
                    #template="plotly_white" 
                )       
        return heatmap
    
def create_pie_chart(labels,values,title):
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    top_labels = sorted_labels[:14]
    top_values = sorted_values[:14]
    
    if len(sorted_values) > 14:
        etc_value = sum(sorted_values[14:])
        top_labels.append("#etc")  
        top_values.append(etc_value)  

    fig = go.Figure(data=[go.Pie(
        labels=top_labels,
        values=top_values,
        hole=0.5,  
        textinfo="percent",
        textposition="inside",
        insidetextorientation="radial", 
        marker=dict(colors=px.colors.qualitative.Set3 + ["#D3D3D3"])  
    )])

    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1, 
            y=0.5,  
            font=dict(size=12)  
        )
    )
    return fig

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


def to_str(value):
    if not isinstance(value, (int, float)): 
        raise ValueError("Only integers and floats are supported")

    units = [("T", 1_000_000_000_000), ("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)]

    for unit, threshold in units:
        if value >= threshold:
            return f"{value / threshold:.1f}{unit}".rstrip("0").rstrip(".") 

    return str(value)