import streamlit as st
import yaml
import os
import pandas as pd
import shutil


def load_yaml(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    return {"credentials": {"usernames": {}}, "preauthorized": {}}

def save_yaml(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
def manager_page():

    YAML_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")

    USER_PATH= os.path.join(os.path.dirname(__file__), "../user")
    data=pd.DataFrame(columns=["Description","url","likes","comments","shared","saved","time"],index=[0])

    st.title("User Information")

    yaml_data = load_yaml(YAML_FILE)
    st.header("운영자 정보")
    current_admin_id = list(yaml_data.get("preauthorized", {}).keys())[0] if yaml_data.get("preauthorized") else ""
    admin_data = yaml_data["preauthorized"].get(current_admin_id, {})

    new_admin_id = st.text_input("운영자 ID", current_admin_id)
    admin_name = st.text_input("운영자 Name", admin_data.get("name", ""))
    admin_pw = st.text_input("운영자 PW", admin_data.get("password", ""), type="password")

    st.header("사용자 목록")
    usernames = yaml_data["credentials"].get("usernames", {})
    edited_users = {}
    users_to_delete = []

    for user_id, user_info in usernames.items():
        with st.expander(f"사용자: {user_id}"):
            user_name = st.text_input(f"Name ({user_id})", user_info.get("name", ""))
            user_pw = st.text_input(f"PW ({user_id})", user_info.get("password", ""), type="password")
            delete_user = st.checkbox(f"삭제 ({user_id})",value=False)
            
            if delete_user:
                users_to_delete.append(user_id)
            else:
                edited_users[user_id] = {"name": user_name, "password": user_pw}

    st.subheader("새 사용자 추가")
    new_user_id = st.text_input("새 사용자 ID",placeholder="")
    new_user_name = st.text_input("새 사용자 Name",placeholder="")
    new_user_pw = st.text_input("새 사용자 PW",placeholder="", type="password")

    if st.button("저장"):
        if new_user_id and new_user_name and new_user_pw:
            edited_users[new_user_id] = {"name": new_user_name, "password": new_user_pw}
            st.success("새 사용자 추가 완료")
            try:
                folder_list=os.listdir(USER_PATH)
                new_folder_list=edited_users.keys()
                
                for user in set(folder_list) - set(new_folder_list):
                    shutil.rmtree(USER_PATH+"/"+user)
                    
                for user in set(new_folder_list):
                    if not os.path.exists(USER_PATH+"/"+user):
                        os.makedirs(USER_PATH+"/"+user, exist_ok=True)
                        data.to_excel(USER_PATH+"/"+user+"/url_list.xlsx", index=False)        
                
                yaml_data["credentials"]["usernames"] = edited_users
                yaml_data["preauthorized"] = {new_admin_id: {"name": admin_name, "password": admin_pw}}
                save_yaml(YAML_FILE, yaml_data)
                st.success("저장됨")

            except yaml.YAMLError as e:
                st.error(f"형식 오류: {e}")

        else:
            try:
                
                folder_list=os.listdir(USER_PATH)
                new_folder_list=edited_users.keys()
                for user in set(folder_list) - set(new_folder_list):
                    shutil.rmtree(USER_PATH+"/"+user)
                    
                for user in set(new_folder_list):
                    if not os.path.exists(USER_PATH+"/"+user):
                        os.makedirs(USER_PATH+"/"+user, exist_ok=True)
                        data.to_excel(USER_PATH+"/"+user+"/url_list.xlsx", index=False)        
        
                    
                    
                yaml_data["credentials"]["usernames"] = edited_users
                yaml_data["preauthorized"] = {new_admin_id: {"name": admin_name, "password": admin_pw}}
                save_yaml(YAML_FILE, yaml_data)
                st.success("저장됨")

            except yaml.YAMLError as e:
                st.error(f"형식 오류: {e}")

            st.error("새 사용자 추가 오류")

        st.rerun()
    if st.button("EXIT"):
        st.session_state['authenticated'] = False

      