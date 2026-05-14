import streamlit as st
from config import TFIDF_DIR, BM25_DIR
from search_engine.rag_model import RAGSearchModel
from search_engine.tfidf_model_svd import TfidfSvdSearchModel
from search_engine.tfidf_model_raw import TfidfRawSearchModel
from search_engine.bm25_model_svd import BM25SvdSearchModel
from search_engine.bm25_model_raw import BM25RawSearchModel


def init_session_state():
    if "active_settings" not in st.session_state:
        st.session_state.active_settings = {
            "option": "RAG",
            "svd": False,
            "n_components": 0,
        }
        st.session_state.model = RAGSearchModel()


def update_model_settings(new_settings):
    st.session_state.active_settings = new_settings.copy()

    if new_settings["option"] == "RAG":
        st.session_state.model = RAGSearchModel()

    elif new_settings["option"] == "TFIDF":
        if not new_settings.get("svd"):
            st.session_state.model = TfidfRawSearchModel()
        else:
            svd_file = TFIDF_DIR / f"tfidf_svd_{new_settings['n_components']}.joblib"
            st.session_state.model = TfidfSvdSearchModel(str(svd_file))

    elif new_settings["option"] == "BM25":
        if not new_settings.get("svd"):
            st.session_state.model = BM25RawSearchModel()
        else:
            svd_file = BM25_DIR / f"bm25_svd_{new_settings['n_components']}.joblib"
            st.session_state.model = BM25SvdSearchModel(str(svd_file))
