import streamlit as st
from frontend.widgets import *
from frontend.session_state import *
from frontend.handle_search import handle_search

init_session_state()


st.title("Gutenberg search engine")
st.text_input("Search for a book", key="search_query")


settings_expander()

settings_status(st.session_state.active_settings)


handle_search()
if "results" in st.session_state:
    st.write("")
    st.write(f"Showing top {len(st.session_state.results)} results:")
    if "search_time" in st.session_state:
        st.write(f"Search took {st.session_state.search_time:.2f} seconds")
    for i, dict_result in enumerate(st.session_state.results):
        display_result(dict_result, i + 1)
