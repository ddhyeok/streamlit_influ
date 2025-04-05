import streamlit as st
import os
import yaml


def authenticate(file_path, username, password):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                
            if config.get("preauthorized", {}).get(username, {}).get("password") == password:
                return "admin"

            if config.get("credentials", {}).get("usernames", {}).get(username, {}).get("password") == password:
                return "main"

        except Exception as e:
            st.error(f"인증 오류 : {str(e)}")
            return False
    else:
        st.error("⚠️ config.yaml 파일을 찾을 수 없습니다.")
        return False

    
def login_page():
    st.title('Login')
    
    username = st.text_input('로그인 ID')
    password = st.text_input('비밀번호', type='password')

    if st.button('로그인'):
        if not username:
            st.error("로그인 ID를 입력하세요.")
        elif not password:
            st.error("비밀번호를 입력하세요.")
        else:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            status = authenticate(file_path=config_path, username=username, password=password)
            if status:
                st.session_state['authenticated'] = status
                st.session_state['user'] = username
                
                st.rerun()

            else:
                st.error('로그인 실패')

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if st.session_state['authenticated']=='admin':
    from manager import manager_page
    manager_page()

elif st.session_state['authenticated']=='main':
    from main import main_page
    main_page()
    
elif st.session_state['authenticated']=='tracking':
    from tracking_page import tracking
    tracking()

elif st.session_state['authenticated']=='ranking':
    from ranking import rank
    rank()

else:
    login_page()