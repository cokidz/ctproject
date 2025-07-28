import streamlit as st
import streamlit.components.v1 as htmlviewer

# Title Msg#1
st.title('This is WebApp!')

with open('./index.html','r', encoding='utf-8') as f:
    html = f.read()
    f.close()

# html = 
# '''
# <html>
#    <head>
#        <title> this is my html </title>
#    </head> 
#    <body>
#        <h1> Topic </h1>
#        <h2> SubTopic </h2>
#    </body>
# </html>

# '''

# Box#1(4), Box#2(1)
col1, col2 = st.columns((4,1))

with col1:
    with st.expander('Ex-Content #1...'):
        url = "https://youtu.be/UTUsmzI0jBA?si=p4LlOE6t1ZsPryED"
        st.info('info-Content..')
        st.video(url)

    with st.expander('Ex-Content #2...'):
        #st.write(html, unsafe_allow_html=True) #html을 파싱하는 코드
        htmlviewer.html(html,height=600)


    with st.expander('Ex-Content #3...'):
        #st.write(html, unsafe_allow_html=True) #html을 파싱하는 코드
        htmlviewer.html(html,height=600)




with col2:
    with st.expander('Tips..'):
        st.info('Tips..')

st.markdown('<hr>', unsafe_allow_html=True)
st.write('<font color="BLUE">(c)Copyright. all rights reserved by DD', unsafe_allow_html=True)