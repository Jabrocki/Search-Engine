import streamlit as st
from config import SVD_COMPONENTS
from frontend.session_state import update_model_settings


def handle_rag_settings():
    st.write("RAG doesn't have any parameters to set!")
    return {}


def handle_svd_settings():
    svd = st.checkbox("Use SVD?")
    print(f"svd: {svd}")
    if svd:
        n_components = st.selectbox(
            "Number of SVD components",
            options=SVD_COMPONENTS,
        )
        return {"svd": True, "n_components": n_components}
    else:

        return {"svd": False}


def settings_expander():
    with st.expander("Search Settings ⚙️", expanded=False):
        st.slider("Number of results", 1, 100, 10, key="num_results")
        ui_option = st.selectbox(
            "Type of model", options=["RAG", "TFIDF", "BM25"], key="ui_model_type"
        )

        ui_settings = {"option": ui_option}
        if ui_option == "RAG":
            ui_settings.update(handle_rag_settings())
        else:
            ui_settings.update(handle_svd_settings())  # type: ignore

        st.button("New settings", on_click=lambda: update_model_settings(ui_settings))


def settings_status(active):
    if active["option"] != "RAG":
        mode_desc = (
            "Using raw matrix without SVD"
            if not active.get("svd")
            else f"Using SVD with {active.get('n_components')} components"
        )
    else:
        mode_desc = "RAG model Active"

    settings_color = "#519C56"
    border_color = "#014F05"

    st.markdown(
        f"""
        <div style="
            background-color: {settings_color};
            border: 2px solid {border_color};
            padding: 15px;
            border-radius: 10px;
            color: {border_color};
        ">
            <strong>Model:</strong> {active['option']}<br>
            {mode_desc}
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_result(dict_result, number):
    main_container = st.container(border=True)
    main_container.markdown(f"### Result {number}")
    main_container.write(f"**{dict_result['book_title']}**")
    if dict_result["book_author"]:
        main_container.write(f"*by {dict_result['book_author']}*")
    col1, col2 = main_container.columns([1, 3])
    col1.image(dict_result["book_image_url"], width="stretch")  # type: ignore
    col2.write("**Text:**")
    col2.write(f"... {dict_result["chunk_text"]} ...")
    main_container.write(f"**Similarity Score:** {dict_result['score']:.4f}")
    main_container.link_button("Go to chunk", url=dict_result["chunk_url"])
