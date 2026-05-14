import streamlit as st
import time


def handle_search():
    if "search_query" not in st.session_state or not st.session_state.search_query:
        return
    query = st.session_state.search_query
    num_results = st.session_state.num_results
    start_time = time.time()
    st.session_state.results = st.session_state.model.search(query, top_n=num_results)
    end_time = time.time()
    st.session_state.search_time = end_time - start_time
