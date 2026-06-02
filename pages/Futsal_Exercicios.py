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

COLOR_SWATCH_OPTIONS = [
    ("#ef4444", "🟥"),
    ("#2563eb", "🟦"),
    ("#eab308", "🟨"),
    ("#22c55e", "🟩"),
    ("#111827", "⬛"),
    ("#ffffff", "⬜"),
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
        div[data-testid="stButton"][data-testid*="exercise_color_toggle"],
        div[data-testid="stButton"][data-testid*="selected_color_toggle"] {
            width: 64px;
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


def _render_saved_exercises() -> None:
    exercises_df = read_exercises()
    if exercises_df.empty:
        st.info("Ainda nao existem exercicios guardados.")
        return
    objects_df = read_exercise_objects()
    work_df = exercises_df.copy()
    work_df["Titulo"] = work_df["titulo"].map(_clean_text)
    work_df["Categoria"] = work_df["categoria"].map(_clean_text)
    work_df["Objetivo"] = work_df["objetivo"].map(_clean_text)
    work_df["Elementos"] = work_df["exercise_id"].map(
        lambda exercise_id: int((objects_df["exercise_id"].astype(str) == str(exercise_id)).sum()) if not objects_df.empty else 0
    )
    st.dataframe(work_df[["Titulo", "Categoria", "Objetivo", "Elementos"]], use_container_width=True, hide_index=True, key="exercise_saved_list")


st.title("Exercicios")
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

meta_cols = st.columns([1.4, 1.1, 1.1], gap="small")
with meta_cols[0]:
    titulo = st.text_input("Titulo", value=_clean_text(exercise_data.get("titulo")), key=f"exercise_title_{selected_exercise_id or 'new'}")
with meta_cols[1]:
    categoria = st.selectbox(
        "Categoria",
        options=CATEGORY_OPTIONS,
        index=CATEGORY_OPTIONS.index(_clean_text(exercise_data.get("categoria"))) if _clean_text(exercise_data.get("categoria")) in CATEGORY_OPTIONS else 0,
        key=f"exercise_category_{selected_exercise_id or 'new'}",
    )
with meta_cols[2]:
    objetivo = st.text_input("Objetivo", value=_clean_text(exercise_data.get("objetivo")), key=f"exercise_goal_{selected_exercise_id or 'new'}")

desc_cols = st.columns([1.3, 1], gap="small")
with desc_cols[0]:
    descricao = st.text_area("Descricao", value=_clean_text(exercise_data.get("descricao")), height=90, key=f"exercise_desc_{selected_exercise_id or 'new'}")
with desc_cols[1]:
    observacoes = st.text_area("Observacoes", value=_clean_text(exercise_data.get("observacoes")), height=90, key=f"exercise_obs_{selected_exercise_id or 'new'}")

palette_mode = st.radio(
    "Separador",
    options=["Objetos", "Formas"],
    horizontal=True,
    key=f"exercise_palette_mode_{selected_exercise_id or 'new'}",
)

default_color = st.session_state.get(f"exercise_color_value_{selected_exercise_id or 'new'}", "#ef4444")

control_cols = st.columns([0.22, 0.55, 3.23], gap="small")
with control_cols[0]:
    st.caption("Cor")
    color_palette_state_key = f"exercise_color_palette_open_{selected_exercise_id or 'new'}"
    if st.button(_swatch_symbol(default_color), key=f"exercise_color_toggle_{selected_exercise_id or 'new'}", use_container_width=False):
        st.session_state[color_palette_state_key] = not st.session_state.get(color_palette_state_key, False)
        st.rerun()
    if st.session_state.get(color_palette_state_key, False):
        color_pick_cols = st.columns(len(COLOR_SWATCH_OPTIONS), gap="small")
        for idx, (color_value, color_symbol) in enumerate(COLOR_SWATCH_OPTIONS):
            if color_pick_cols[idx].button(color_symbol, key=f"exercise_color_pick_{selected_exercise_id or 'new'}_{idx}", use_container_width=True):
                st.session_state[f"exercise_color_value_{selected_exercise_id or 'new'}"] = color_value
                st.session_state[color_palette_state_key] = False
                st.rerun()
element_color = st.session_state.get(f"exercise_color_value_{selected_exercise_id or 'new'}", "#ef4444")
element_label = control_cols[1].text_input("Rotulo", value="", key=f"exercise_label_{selected_exercise_id or 'new'}")
st.caption("Arrasta o objeto da lateral para o campo. Depois podes arrastar os objetos dentro do proprio campo.")

board_state = tactical_board(
    field_image_url=_field_image_url(),
    objects=board_objects,
    active_color=element_color,
    active_size=DEFAULT_OBJECT_SIZE,
    active_label=_clean_text(element_label),
    exercise_id=selected_exercise_id or "new",
    objects_revision=board_revision,
    palette_mode="shapes" if palette_mode == "Formas" else "objects",
    height=650,
    key=f"tactical_board_{selected_exercise_id or 'new'}",
)
if board_state and isinstance(board_state, dict):
    st.session_state["exercise_board_objects"] = board_state.get("objects", board_objects)
    board_objects = st.session_state["exercise_board_objects"]
    st.session_state["exercise_selected_object_id"] = board_state.get("selected_id")

selected_object_id = _clean_text(st.session_state.get("exercise_selected_object_id"))
selected_object = next((obj for obj in board_objects if _clean_text(obj.get("id")) == selected_object_id), None)

if selected_object:
    st.markdown("### Objeto selecionado")
    selected_cols = st.columns([1.15, 0.55, 0.22], gap="small")
    selected_cols[0].text_input(
        "Tipo",
        value=_clean_text(selected_object.get("type")),
        disabled=True,
        key=f"selected_type_{selected_exercise_id or 'new'}",
    )
    selected_label = selected_cols[1].text_input(
        "Rotulo",
        value=_clean_text(selected_object.get("label")),
        key=f"selected_label_{selected_exercise_id or 'new'}",
    )
    selected_color_key = f"selected_color_value_{selected_exercise_id or 'new'}"
    current_selected_color = _clean_text(selected_object.get("color")) or "#ef4444"
    if st.session_state.get(selected_color_key) != current_selected_color:
        st.session_state[selected_color_key] = current_selected_color
    with selected_cols[2]:
        st.caption("Cor")
        selected_palette_state_key = f"selected_color_palette_open_{selected_exercise_id or 'new'}"
        if st.button(_swatch_symbol(st.session_state[selected_color_key]), key=f"selected_color_toggle_{selected_exercise_id or 'new'}", use_container_width=False):
            st.session_state[selected_palette_state_key] = not st.session_state.get(selected_palette_state_key, False)
            st.rerun()
        if st.session_state.get(selected_palette_state_key, False):
            selected_pick_cols = st.columns(len(COLOR_SWATCH_OPTIONS), gap="small")
            for idx, (color_value, color_symbol) in enumerate(COLOR_SWATCH_OPTIONS):
                if selected_pick_cols[idx].button(color_symbol, key=f"selected_color_pick_{selected_exercise_id or 'new'}_{idx}", use_container_width=True):
                    st.session_state[selected_color_key] = color_value
                    st.session_state[selected_palette_state_key] = False
                    st.rerun()
    selected_color = st.session_state[selected_color_key]
    if st.button("Atualizar objeto selecionado", use_container_width=False, key=f"update_selected_{selected_exercise_id or 'new'}"):
        updated_objects = []
        for obj in board_objects:
            if _clean_text(obj.get("id")) == selected_object_id:
                updated = dict(obj)
                updated["label"] = _clean_text(selected_label)
                updated["color"] = selected_color
                updated_objects.append(updated)
            else:
                updated_objects.append(obj)
        st.session_state["exercise_board_objects"] = updated_objects
        st.session_state["exercise_board_revision"] = int(st.session_state.get("exercise_board_revision", 0)) + 1
        st.rerun()

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

st.markdown("### Biblioteca de exercicios")
_render_saved_exercises()
