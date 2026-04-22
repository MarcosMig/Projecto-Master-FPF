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
        "description": "Area reservada para testes e futura configuracao.",
        "status": "preview",
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
    requested_modality = st.query_params.get("modalidade", "")
    if isinstance(requested_modality, list):
        requested_modality = requested_modality[0] if requested_modality else ""
    requested_modality = str(requested_modality or "").strip()
    if requested_modality in MODALITIES:
        select_modality(requested_modality)
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.rerun()

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
            display: flex;
            justify-content: center;
            align-items: stretch;
            gap: 1rem;
            width: 100%;
            margin-top: 2.5rem;
          }
          .modality-card {
            width: 260px;
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
            text-decoration: none;
            padding: 1.25rem;
          }
          .modality-card:hover {
            border-color: #E30613;
            color: #ffffff;
            background: #1f2732;
            text-decoration: none;
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

    cards_html = ['<div class="modality-grid">']
    for key, modality in MODALITIES.items():
        cards_html.append(
            f"""
            <a class="modality-card" href="?modalidade={key}" target="_self">
              <div class="modality-card-title">{modality["label"]}</div>
              <div class="modality-card-description">{modality["description"]}</div>
            </a>
            """
        )
    cards_html.append("</div>")
    st.html("".join(cards_html))


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
