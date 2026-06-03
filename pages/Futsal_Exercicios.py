from __future__ import annotations

import json
import base64
import io
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

from fpf_modules.futsal_exercise_store import (
    delete_exercise,
    read_exercise_objects,
    read_exercises,
    upsert_exercise,
)
from fpf_modules.tactical_board_component import tactical_board


CATEGORY_OPTIONS = [
    "",
    "Ativacao",
    "Tecnica Individual",
    "Tatica",
    "Organizacao Ofensiva",
    "Organizacao Defensiva",
    "Transicao",
    "Finalizacao",
    "Jogo Condicionado",
    "Competicao",
]

ESPACO_OPTIONS = [
    "",
    "Campo Inteiro",
    "Meio Campo",
    "3/4 Campo",
    "Campo reduzido",
    "Fora de Campo",
]

FORMA_OPTIONS = [
    "",
    "Fundamental",
    "Condicionada",
    "Competitiva",
    "Analitica",
    "Integrada",
]

CAPACIDADE_MOTORA_OPTIONS = [
    "",
    "Forca",
    "Forca Resistente",
    "Velocidade",
    "Resistencia",
    "Resistencia Especifica",
    "Coordenacao",
    "Mobilidade",
]

INTENSIDADE_OPTIONS = [
    "",
    "Baixa",
    "Media",
    "Alta",
    "Maxima",
]

RECUPERACAO_OPTIONS = [
    "",
    "Completa",
    "Incompleta",
    "Ativa",
    "Passiva",
]

VOLUME_OPTIONS = [
    "",
    "Baixo",
    "Medio",
    "Alto",
]

COLOR_SWATCH_OPTIONS = [
    ("#ef4444", "🟥"),
    ("#2563eb", "🟦"),
    ("#eab308", "🟨"),
    ("#f97316", "🟧"),
    ("#22c55e", "🟩"),
    ("#ff2d2d", "🟥"),
]


def _swatch_symbol(color_value: str) -> str:
    normalized = _clean_text(color_value).lower() or "#ef4444"
    for color, symbol in COLOR_SWATCH_OPTIONS:
        if color.lower() == normalized:
            return symbol
    return "⬛"


def _set_swatch_button_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] button[kind="secondary"] {
            min-height: 56px;
            border-radius: 14px;
        }
        div[data-testid="stButton"] button[kind="secondary"][id*="exercise_color_toggle"],
        div[data-testid="stButton"] button[kind="secondary"][id*="selected_color_toggle"] {
            width: 84px;
            min-width: 84px;
            max-width: 84px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

CANVAS_WIDTH = 900
CANVAS_HEIGHT = 560
DEFAULT_OBJECT_SIZE = 22
FIELD_IMAGE_PATH = Path(__file__).resolve().parents[1] / "assets" / "futsal" / "FtsField.png"

def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _build_pitch_background() -> Image.Image:
    if FIELD_IMAGE_PATH.exists():
        return Image.open(FIELD_IMAGE_PATH).convert("RGB").resize((CANVAS_WIDTH, CANVAS_HEIGHT))
    return Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "#d4d4d8")


def _field_image_url() -> str:
    buffer = io.BytesIO()
    _build_pitch_background().save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _x_to_pct(x_px: float) -> float:
    return round(max(0.0, min(100.0, (x_px / CANVAS_WIDTH) * 100)), 2)


def _y_to_pct(y_px: float) -> float:
    return round(max(0.0, min(100.0, (y_px / CANVAS_HEIGHT) * 100)), 2)


def _pct_to_x(x_pct: float) -> float:
    return max(0.0, min(CANVAS_WIDTH, (float(x_pct) / 100.0) * CANVAS_WIDTH))


def _pct_to_y(y_pct: float) -> float:
    return max(0.0, min(CANVAS_HEIGHT, (float(y_pct) / 100.0) * CANVAS_HEIGHT))


def _objects_to_rows(objects: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, obj in enumerate(objects, start=1):
        rows.append(
            {
                "object_id": _clean_text(obj.get("id")),
                "ordem": idx,
                "tipo": _clean_text(obj.get("type")) or "Jogador",
                "rotulo": _clean_text(obj.get("label")),
                "cor": _clean_text(obj.get("color")) or "#ef4444",
                "x_pct": float(obj.get("x_pct") or 50),
                "y_pct": float(obj.get("y_pct") or 50),
                "tamanho": int(float(obj.get("size") or 18)),
            }
        )
    return pd.DataFrame(rows, columns=["object_id", "ordem", "tipo", "rotulo", "cor", "x_pct", "y_pct", "tamanho"])


def _rows_to_component_objects(objects_df: pd.DataFrame) -> list[dict[str, Any]]:
    if objects_df.empty:
        return []
    objects: list[dict[str, Any]] = []
    for _, row in objects_df.iterrows():
        objects.append(
            {
                "id": _clean_text(row.get("object_id")),
                "type": _clean_text(row.get("tipo")) or "Jogador",
                "label": _clean_text(row.get("rotulo")),
                "color": _clean_text(row.get("cor")) or "#ef4444",
                "x_pct": float(row.get("x_pct") or 50),
                "y_pct": float(row.get("y_pct") or 50),
                "size": int(float(row.get("tamanho") or 18)),
            }
        )
    return objects


def _next_object_id() -> str:
    import uuid

    return uuid.uuid4().hex


def _load_exercise_state(selected_exercise_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exercises_df = read_exercises()
    exercise_row = exercises_df[exercises_df["exercise_id"].astype(str) == str(selected_exercise_id)].head(1)
    if exercise_row.empty:
        return {}, []
    exercise_data = exercise_row.iloc[0].to_dict()
    objects_df = read_exercise_objects()
    exercise_objects = objects_df[objects_df["exercise_id"].astype(str) == str(selected_exercise_id)].copy()
    return exercise_data, _rows_to_component_objects(exercise_objects)


def _option_index(options: list[str], value: Any) -> int:
    cleaned = _clean_text(value)
    return options.index(cleaned) if cleaned in options else 0


st.title("Criar Exercicio")
st.caption("Campo base fixo com elementos desenhados por cima.")
_set_swatch_button_styles()

exercises_df = read_exercises()
exercise_options = [("", "Novo exercicio")]
for _, row in exercises_df.sort_values("updated_at", ascending=False, na_position="last").iterrows():
    label = _clean_text(row.get("titulo")) or _clean_text(row.get("exercise_id"))
    exercise_options.append((_clean_text(row.get("exercise_id")), label))

selected_exercise_id = st.selectbox(
    "Exercicio",
    options=[option[0] for option in exercise_options],
    format_func=lambda value: next((label for key, label in exercise_options if key == value), value or "Novo exercicio"),
    key="exercise_selector",
)

loaded_marker = st.session_state.get("exercise_loaded_id")
if loaded_marker != selected_exercise_id:
    loaded_data, loaded_canvas = _load_exercise_state(selected_exercise_id)
    st.session_state["exercise_loaded_id"] = selected_exercise_id
    st.session_state["exercise_form_data"] = loaded_data
    st.session_state["exercise_board_objects"] = loaded_canvas
    st.session_state["exercise_board_revision"] = int(st.session_state.get("exercise_board_revision", 0)) + 1

exercise_data = st.session_state.get("exercise_form_data", {})
board_objects = st.session_state.get("exercise_board_objects", [])
board_revision = int(st.session_state.get("exercise_board_revision", 0))

head_row_1 = st.columns([1.25, 1], gap="small")
with head_row_1[0]:
    titulo = st.text_input("Titulo do Exercicio", value=_clean_text(exercise_data.get("titulo")), key=f"exercise_title_{selected_exercise_id or 'new'}")
with head_row_1[1]:
    objetivo = st.text_input("Objetivo", value=_clean_text(exercise_data.get("objetivo")), key=f"exercise_goal_{selected_exercise_id or 'new'}")

head_row_2 = st.columns([1, 1], gap="small")
with head_row_2[0]:
    contexto = st.text_input("Contexto", value=_clean_text(exercise_data.get("contexto")), key=f"exercise_context_{selected_exercise_id or 'new'}")
with head_row_2[1]:
    estrutura_funcional = st.text_input(
        "Estrutura Funcional",
        value=_clean_text(exercise_data.get("estrutura_funcional")),
        key=f"exercise_structure_{selected_exercise_id or 'new'}",
    )

head_row_3 = st.columns([1, 0.8, 0.9, 1], gap="small")
with head_row_3[0]:
    espaco = st.selectbox(
        "Espaco",
        options=ESPACO_OPTIONS,
        index=_option_index(ESPACO_OPTIONS, exercise_data.get("espaco")),
        key=f"exercise_space_{selected_exercise_id or 'new'}",
    )
with head_row_3[1]:
    numero_jogadores = st.text_input(
        "Numero de Jogadores",
        value=_clean_text(exercise_data.get("numero_jogadores")),
        key=f"exercise_players_{selected_exercise_id or 'new'}",
    )
with head_row_3[2]:
    forma = st.selectbox(
        "Forma",
        options=FORMA_OPTIONS,
        index=_option_index(FORMA_OPTIONS, exercise_data.get("forma")),
        key=f"exercise_shape_{selected_exercise_id or 'new'}",
    )
with head_row_3[3]:
    categoria = st.selectbox(
        "Categoria",
        options=CATEGORY_OPTIONS,
        index=_option_index(CATEGORY_OPTIONS, exercise_data.get("categoria")),
        key=f"exercise_category_{selected_exercise_id or 'new'}",
    )

head_row_4 = st.columns([1, 0.8, 0.8], gap="small")
with head_row_4[0]:
    capacidade_motora = st.selectbox(
        "Capacidade Motora",
        options=CAPACIDADE_MOTORA_OPTIONS,
        index=_option_index(CAPACIDADE_MOTORA_OPTIONS, exercise_data.get("capacidade_motora")),
        key=f"exercise_motor_{selected_exercise_id or 'new'}",
    )
with head_row_4[1]:
    duracao = st.text_input("Duracao", value=_clean_text(exercise_data.get("duracao")), key=f"exercise_duration_{selected_exercise_id or 'new'}")
with head_row_4[2]:
    intensidade = st.selectbox(
        "Intensidade",
        options=INTENSIDADE_OPTIONS,
        index=_option_index(INTENSIDADE_OPTIONS, exercise_data.get("intensidade")),
        key=f"exercise_intensity_{selected_exercise_id or 'new'}",
    )

pedagogic_cols = st.columns(3, gap="small")
with pedagogic_cols[0]:
    condicionantes = st.text_area(
        "Condicionantes",
        value=_clean_text(exercise_data.get("condicionantes")),
        height=90,
        key=f"exercise_constraints_{selected_exercise_id or 'new'}",
    )
with pedagogic_cols[1]:
    comportamentos_ofensivos = st.text_area(
        "Comportamentos Ofensivos",
        value=_clean_text(exercise_data.get("comportamentos_ofensivos")),
        height=90,
        key=f"exercise_offensive_{selected_exercise_id or 'new'}",
    )
with pedagogic_cols[2]:
    comportamentos_defensivos = st.text_area(
        "Comportamentos Defensivos",
        value=_clean_text(exercise_data.get("comportamentos_defensivos")),
        height=90,
        key=f"exercise_defensive_{selected_exercise_id or 'new'}",
    )

load_cols = st.columns(4, gap="small")
with load_cols[0]:
    frequencia = st.text_input("Frequencia", value=_clean_text(exercise_data.get("frequencia")), key=f"exercise_frequency_{selected_exercise_id or 'new'}")
with load_cols[1]:
    recuperacao = st.selectbox(
        "Recuperacao",
        options=RECUPERACAO_OPTIONS,
        index=_option_index(RECUPERACAO_OPTIONS, exercise_data.get("recuperacao")),
        key=f"exercise_recovery_{selected_exercise_id or 'new'}",
    )
with load_cols[2]:
    densidade = st.text_input("Densidade", value=_clean_text(exercise_data.get("densidade")), key=f"exercise_density_{selected_exercise_id or 'new'}")
with load_cols[3]:
    volume = st.selectbox(
        "Volume",
        options=VOLUME_OPTIONS,
        index=_option_index(VOLUME_OPTIONS, exercise_data.get("volume")),
        key=f"exercise_volume_{selected_exercise_id or 'new'}",
    )

desc_cols = st.columns([1.4, 1], gap="small")
with desc_cols[0]:
    descricao = st.text_area("Descricao do Exercicio", value=_clean_text(exercise_data.get("descricao")), height=100, key=f"exercise_desc_{selected_exercise_id or 'new'}")
with desc_cols[1]:
    observacoes = st.text_area("Observacoes", value=_clean_text(exercise_data.get("observacoes")), height=100, key=f"exercise_obs_{selected_exercise_id or 'new'}")

st.caption("Arrasta o objeto da lateral para o campo. Depois podes arrastar os objetos dentro do proprio campo.")

element_label = ""
element_color = "#ef4444"

board_state = tactical_board(
    field_image_url=_field_image_url(),
    objects=board_objects,
    active_color=element_color,
    active_size=DEFAULT_OBJECT_SIZE,
    active_label=_clean_text(element_label),
    exercise_id=selected_exercise_id or "new",
    objects_revision=board_revision,
    height=650,
    key=f"tactical_board_{selected_exercise_id or 'new'}",
)
if board_state and isinstance(board_state, dict):
    st.session_state["exercise_board_objects"] = board_state.get("objects", board_objects)
    board_objects = st.session_state["exercise_board_objects"]
    st.session_state["exercise_selected_object_id"] = board_state.get("selected_id")

selected_object_id = _clean_text(st.session_state.get("exercise_selected_object_id"))
selected_object = next((obj for obj in board_objects if _clean_text(obj.get("id")) == selected_object_id), None)

action_cols = st.columns([1.05, 1.05, 1.05, 1.05, 1.05, 2.75], gap="small")
if action_cols[0].button("Desfazer ultimo", use_container_width=True):
    work = list(st.session_state.get("exercise_board_objects", []))
    st.session_state["exercise_board_objects"] = work[:-1]
    st.session_state["exercise_board_revision"] = int(st.session_state.get("exercise_board_revision", 0)) + 1
    st.rerun()
if action_cols[1].button("Limpar campo", use_container_width=True):
    st.session_state["exercise_board_objects"] = []
    st.session_state["exercise_board_revision"] = int(st.session_state.get("exercise_board_revision", 0)) + 1
    st.rerun()
if action_cols[2].button("Apagar selecionado", use_container_width=True, disabled=not selected_object_id):
    current_objects = list(st.session_state.get("exercise_board_objects", []))
    st.session_state["exercise_board_objects"] = [obj for obj in current_objects if _clean_text(obj.get("id")) != selected_object_id]
    st.session_state["exercise_selected_object_id"] = ""
    st.session_state["exercise_board_revision"] = int(st.session_state.get("exercise_board_revision", 0)) + 1
    st.rerun()
if action_cols[3].button("Duplicar selecionado", use_container_width=True, disabled=not selected_object_id):
    current_objects = list(st.session_state.get("exercise_board_objects", []))
    duplicate_source = next((obj for obj in current_objects if _clean_text(obj.get("id")) == selected_object_id), None)
    if duplicate_source:
        duplicated = dict(duplicate_source)
        duplicated["id"] = _next_object_id()
        duplicated["x_pct"] = min(98.0, float(duplicated.get("x_pct", 50)) + 4.0)
        duplicated["y_pct"] = min(98.0, float(duplicated.get("y_pct", 50)) + 4.0)
        st.session_state["exercise_board_objects"] = [*current_objects, duplicated]
        st.session_state["exercise_selected_object_id"] = duplicated["id"]
        st.session_state["exercise_board_revision"] = int(st.session_state.get("exercise_board_revision", 0)) + 1
        st.rerun()
if action_cols[4].button("Gravar exercicio", type="primary", use_container_width=True):
    try:
        if not _clean_text(titulo):
            raise RuntimeError("O titulo do exercicio e obrigatorio.")
        current_objects = list(st.session_state.get("exercise_board_objects", []))
        objects_df = _objects_to_rows(current_objects)
        if objects_df.empty:
            raise RuntimeError("Adiciona pelo menos um elemento ao campo antes de gravar o exercicio.")
        exercise_id = upsert_exercise(
            {
                "exercise_id": selected_exercise_id,
                  "titulo": _clean_text(titulo),
                  "categoria": _clean_text(categoria),
                  "objetivo": _clean_text(objetivo),
                  "contexto": _clean_text(contexto),
                  "estrutura_funcional": _clean_text(estrutura_funcional),
                  "espaco": _clean_text(espaco),
                  "numero_jogadores": _clean_text(numero_jogadores),
                  "forma": _clean_text(forma),
                  "capacidade_motora": _clean_text(capacidade_motora),
                  "duracao": _clean_text(duracao),
                  "intensidade": _clean_text(intensidade),
                  "condicionantes": _clean_text(condicionantes),
                  "comportamentos_ofensivos": _clean_text(comportamentos_ofensivos),
                  "comportamentos_defensivos": _clean_text(comportamentos_defensivos),
                  "frequencia": _clean_text(frequencia),
                  "recuperacao": _clean_text(recuperacao),
                  "densidade": _clean_text(densidade),
                  "volume": _clean_text(volume),
                  "descricao": _clean_text(descricao),
                  "observacoes": _clean_text(observacoes),
                  "canvas_json": json.dumps({"objects": current_objects}),
            },
            objects_df,
        )
    except Exception as exc:
        st.error(str(exc))
    else:
        st.session_state["exercise_editor_success"] = f"Exercicio gravado com sucesso. ID: {exercise_id}"
        st.session_state["exercise_loaded_id"] = None
        st.rerun()
if selected_exercise_id and action_cols[5].button("Eliminar exercicio", use_container_width=True):
    delete_exercise(selected_exercise_id)
    st.session_state["exercise_board_objects"] = []
    st.session_state["exercise_loaded_id"] = None
    st.session_state["exercise_form_data"] = {}
    st.session_state["exercise_board_revision"] = int(st.session_state.get("exercise_board_revision", 0)) + 1
    st.session_state["exercise_editor_success"] = "Exercicio eliminado com sucesso."
    st.rerun()

success_message = st.session_state.pop("exercise_editor_success", "")
if success_message:
    st.success(success_message)
