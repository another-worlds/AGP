import streamlit as st
import langchain_helper as lch
import textwrap


# video_url = "https://www.youtube.com/watch?v=lG7Uxts9SXs"
st.set_page_config(initial_sidebar_state="expanded")


st.title("Shapefile assistant")
with st.sidebar:
    with st.form(key="my_form"):
        zone_id = st.sidebar.text_area(
            label="What is your question",
            max_chars=50
        )
        
        submit_button = st.form_submit_button(label="Submit")
        
if submit_button:
    response = lch.run_multi_agent(zone_id)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))
    
    