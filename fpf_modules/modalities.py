import streamlit as st


MODALITIES = {
    "futebol": {
        "label": "Futebol",
        "title": "FPF Performance Hub",
        "description": "Analise GPS, atletas, campos, jogos e treinos.",
        "status": "active",
    },
    "futsal": {
        "label": "Futsal",
        "title": "Futsal Performance Hub",
        "description": "Base de dados de atletas e avaliacao integrada da modalidade.",
        "status": "active",
    },
}


def ensure_modality_state() -> None:
    if "selected_modality" not in st.session_state:
        st.session_state.selected_modality = ""


def select_modality(modality_key: str) -> None:
    if modality_key in MODALITIES:
        st.session_state.selected_modality = modality_key


def clear_modality() -> None:
    st.session_state.pop("selected_modality", None)
    try:
        st.query_params.clear()
    except Exception:
        pass
    ensure_modality_state()


def current_modality() -> dict:
    ensure_modality_state()
    return MODALITIES.get(st.session_state.selected_modality, {})


def current_modality_label() -> str:
    return str(current_modality().get("label") or "")


def render_modality_selector() -> None:
    ensure_modality_state()

    st.markdown(
        """
        <style>
          .stApp { background-color: #0e1117; }
          [data-testid="stSidebar"] {display: none;}
          .main .block-container {
            padding-top: 12vh !important;
            max-width: 720px !important;
            margin-left: auto !important;
            margin-right: auto !important;
          }
          .modality-title {
            color: #ffffff;
            font-size: 2rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.35rem;
          }
          .modality-subtitle {
            color: #9aa0a6;
            text-align: center;
            margin-bottom: 2rem;
          }
          .modality-grid {
            margin-top: 2.5rem;
          }
          .modality-card {
            width: 100%;
            min-height: 12.5rem;
            border-radius: 12px;
            border: 1px solid #30363d;
            background: #151b23;
            color: #ffffff;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.22);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 1.25rem;
          }
          .modality-card-active {
            border-color: #E30613;
            background: #1f2732;
          }
          .modality-button .stButton > button {
            width: 100%;
            min-height: 12.5rem;
            border-radius: 12px;
            border: 1px solid #30363d;
            background: transparent !important;
            color: transparent !important;
            box-shadow: none !important;
          }
          .modality-button .stButton > button:hover {
            border-color: #E30613 !important;
            background: rgba(227, 6, 19, 0.06) !important;
          }
          .modality-card-title {
            font-size: 1.35rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
          }
          .modality-card-description {
            color: #9aa0a6;
            font-size: 0.95rem;
            line-height: 1.35;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="modality-title">Escolher modalidade</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="modality-subtitle">Seleciona o contexto de trabalho para abrir o Hub correto.</div>',
        unsafe_allow_html=True,
    )
    left_col, right_col = st.columns(2, gap="large")
    columns = [left_col, right_col]

    for index, (key, modality) in enumerate(MODALITIES.items()):
        with columns[index % len(columns)]:
            st.markdown('<div class="modality-button">', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="modality-card">
                  <div class="modality-card-title">{modality["label"]}</div>
                  <div class="modality-card-description">{modality["description"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Abrir {modality['label']}", key=f"select_modality_{key}", type="primary"):
                select_modality(key)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def render_preview_modality() -> None:
    modality = current_modality()
    st.title(modality.get("title", "Performance Hub"))
    st.info(
        "Esta modalidade esta isolada para testes. "
        "As paginas e analises de Futebol nao sao carregadas neste contexto."
    )
    if st.button("Voltar a escolha de modalidade"):
        clear_modality()
        st.rerun()
