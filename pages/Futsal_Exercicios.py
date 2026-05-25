from __future__ import annotations

import json
import uuid
import base64
import io
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
import streamlit.elements.image as st_image
from streamlit_drawable_canvas import st_canvas

from fpf_modules.futsal_exercise_store import (
    delete_exercise,
    read_exercise_objects,
    read_exercises,
    upsert_exercise,
)


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

OBJECT_TYPE_OPTIONS = ["Jogador", "Cone", "Sinalizador", "Baliza"]
CANVAS_WIDTH = 900
CANVAS_HEIGHT = 560
FIELD_LEFT = 70
FIELD_RIGHT = CANVAS_WIDTH - 70
FIELD_TOP = 40
FIELD_BOTTOM = CANVAS_HEIGHT - 40


if not hasattr(st_image, "image_to_url"):
    def _image_to_url_compat(
        image: Image.Image,
        width: int | None = None,
        clamp: bool | None = None,
        channels: str | None = None,
        output_format: str = "PNG",
        image_id: str | None = None,
    ) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format=output_format)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        mime = f"image/{output_format.lower()}"
        return f"data:{mime};base64,{encoded}"

    st_image.image_to_url = _image_to_url_compat


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _blank_canvas() -> dict[str, Any]:
    return {"version": "4.4.0", "objects": [], "background": "rgba(0,0,0,0)"}


def _build_pitch_background() -> Image.Image:
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "#0f172a")
    draw = ImageDraw.Draw(img)
    field_fill = "#15803d"
    draw.rounded_rectangle(
        [(0, 0), (CANVAS_WIDTH - 1, CANVAS_HEIGHT - 1)],
        radius=28,
        fill="#111827",
    )
    draw.rounded_rectangle(
        [(18, 18), (CANVAS_WIDTH - 18, CANVAS_HEIGHT - 18)],
        radius=26,
        fill=field_fill,
    )
    draw.rounded_rectangle(
        [(FIELD_LEFT, FIELD_TOP), (FIELD_RIGHT, FIELD_BOTTOM)],
        radius=8,
        outline="#ffffff",
        width=5,
    )
    center_x = CANVAS_WIDTH // 2
    center_y = CANVAS_HEIGHT // 2
    draw.line([(center_x, FIELD_TOP), (center_x, FIELD_BOTTOM)], fill="#ffffff", width=4)
    draw.ellipse(
        [(center_x - 54, center_y - 54), (center_x + 54, center_y + 54)],
        outline="#ffffff",
        width=4,
    )
    draw.ellipse(
        [(center_x - 5, center_y - 5), (center_x + 5, center_y + 5)],
        fill="#ffffff",
    )
    draw.rectangle([(FIELD_LEFT, center_y - 88), (FIELD_LEFT + 122, center_y + 88)], outline="#ffffff", width=4)
    draw.rectangle([(FIELD_RIGHT - 122, center_y - 88), (FIELD_RIGHT, center_y + 88)], outline="#ffffff", width=4)
    draw.rectangle([(FIELD_LEFT - 10, center_y - 28), (FIELD_LEFT, center_y + 28)], fill="#ffffff")
    draw.rectangle([(FIELD_RIGHT, center_y - 28), (FIELD_RIGHT + 10, center_y + 28)], fill="#ffffff")
    return img


def _x_to_pct(x_px: float) -> float:
    return round(max(0.0, min(100.0, (x_px / CANVAS_WIDTH) * 100)), 2)


def _y_to_pct(y_px: float) -> float:
    return round(max(0.0, min(100.0, (y_px / CANVAS_HEIGHT) * 100)), 2)


def _pct_to_x(x_pct: float) -> float:
    return max(0.0, min(CANVAS_WIDTH, (float(x_pct) / 100.0) * CANVAS_WIDTH))


def _pct_to_y(y_pct: float) -> float:
    return max(0.0, min(CANVAS_HEIGHT, (float(y_pct) / 100.0) * CANVAS_HEIGHT))


def _fabric_object(
    element_type: str,
    x_px: float,
    y_px: float,
    color: str,
    size: int,
    object_id: str | None = None,
    label: str = "",
) -> dict[str, Any]:
    object_id = object_id or uuid.uuid4().hex
    base = {
        "left": float(x_px),
        "top": float(y_px),
        "originX": "center",
        "originY": "center",
        "stroke": "#ffffff",
        "strokeWidth": 2,
        "objectCaching": False,
        "transparentCorners": False,
        "cornerColor": "#22c55e",
        "cornerStrokeColor": "#ffffff",
        "borderColor": "#22c55e",
        "object_id": object_id,
        "element_type": element_type,
        "label": label,
        "size_ref": int(size),
    }
    if element_type == "Jogador":
        return {
            **base,
            "type": "circle",
            "radius": int(size),
            "fill": color,
        }
    if element_type == "Cone":
        return {
            **base,
            "type": "triangle",
            "width": int(size * 1.9),
            "height": int(size * 1.9),
            "fill": color,
        }
    if element_type == "Sinalizador":
        return {
            **base,
            "type": "rect",
            "width": int(size * 1.6),
            "height": int(size * 1.6),
            "angle": 45,
            "fill": color,
        }
    return {
        **base,
        "type": "rect",
        "width": int(size * 3.0),
        "height": int(size * 1.5),
        "fill": "rgba(255,255,255,0.03)",
        "stroke": color,
        "strokeWidth": 4,
    }


def _ensure_canvas_ids(canvas_json: dict[str, Any]) -> dict[str, Any]:
    work = json.loads(json.dumps(canvas_json or _blank_canvas()))
    for obj in work.get("objects", []):
        if not _clean_text(obj.get("object_id")):
            obj["object_id"] = uuid.uuid4().hex
        if not _clean_text(obj.get("element_type")):
            obj["element_type"] = _shape_to_element_type(_clean_text(obj.get("type")))
        if "size_ref" not in obj:
            obj["size_ref"] = _shape_size(obj)
    return work


def _shape_to_element_type(shape_type: str) -> str:
    mapping = {
        "circle": "Jogador",
        "triangle": "Cone",
        "rect": "Sinalizador",
    }
    return mapping.get(shape_type, "Jogador")


def _shape_size(obj: dict[str, Any]) -> int:
    if _clean_text(obj.get("element_type")) == "Jogador":
        return int(float(obj.get("radius") or 18))
    width = float(obj.get("width") or 18)
    return int(max(8, round(width / 1.6)))


def _objects_to_rows(canvas_json: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    objects = canvas_json.get("objects", []) if canvas_json else []
    for idx, obj in enumerate(objects, start=1):
        element_type = _clean_text(obj.get("element_type")) or _shape_to_element_type(_clean_text(obj.get("type")))
        x_px = float(obj.get("left") or 0)
        y_px = float(obj.get("top") or 0)
        color = _clean_text(obj.get("fill")) or _clean_text(obj.get("stroke")) or "#ef4444"
        if element_type == "Baliza":
            color = _clean_text(obj.get("stroke")) or "#ef4444"
        rows.append(
            {
                "object_id": _clean_text(obj.get("object_id")) or uuid.uuid4().hex,
                "ordem": idx,
                "tipo": element_type,
                "rotulo": _clean_text(obj.get("label")),
                "cor": color,
                "x_pct": _x_to_pct(x_px),
                "y_pct": _y_to_pct(y_px),
                "tamanho": int(obj.get("size_ref") or _shape_size(obj)),
            }
        )
    return pd.DataFrame(rows, columns=["object_id", "ordem", "tipo", "rotulo", "cor", "x_pct", "y_pct", "tamanho"])


def _rows_to_canvas(objects_df: pd.DataFrame) -> dict[str, Any]:
    if objects_df.empty:
        return _blank_canvas()
    objects: list[dict[str, Any]] = []
    for _, row in objects_df.iterrows():
        obj = _fabric_object(
            element_type=_clean_text(row.get("tipo")) or "Jogador",
            x_px=_pct_to_x(float(row.get("x_pct") or 50)),
            y_px=_pct_to_y(float(row.get("y_pct") or 50)),
            color=_clean_text(row.get("cor")) or "#ef4444",
            size=int(float(row.get("tamanho") or 18)),
            object_id=_clean_text(row.get("object_id")) or None,
            label=_clean_text(row.get("rotulo")),
        )
        objects.append(obj)
    return {"version": "4.4.0", "objects": objects, "background": "rgba(0,0,0,0)"}


def _load_exercise_state(selected_exercise_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    exercises_df = read_exercises()
    exercise_row = exercises_df[exercises_df["exercise_id"].astype(str) == str(selected_exercise_id)].head(1)
    if exercise_row.empty:
        return {}, _blank_canvas()
    exercise_data = exercise_row.iloc[0].to_dict()
    canvas_json_raw = _clean_text(exercise_data.get("canvas_json"))
    if canvas_json_raw:
        try:
            return exercise_data, _ensure_canvas_ids(json.loads(canvas_json_raw))
        except json.JSONDecodeError:
            pass
    objects_df = read_exercise_objects()
    exercise_objects = objects_df[objects_df["exercise_id"].astype(str) == str(selected_exercise_id)].copy()
    return exercise_data, _rows_to_canvas(exercise_objects)


def _promote_new_points(
    raw_canvas: dict[str, Any],
    element_type: str,
    color: str,
    size: int,
    label: str,
) -> tuple[dict[str, Any], bool]:
    work = json.loads(json.dumps(raw_canvas or _blank_canvas()))
    changed = False
    promoted_objects: list[dict[str, Any]] = []
    for obj in work.get("objects", []):
        if _clean_text(obj.get("element_type")):
            promoted_objects.append(obj)
            continue
        x_px = float(obj.get("left") or 0)
        y_px = float(obj.get("top") or 0)
        promoted_objects.append(
            _fabric_object(
                element_type=element_type,
                x_px=x_px,
                y_px=y_px,
                color=color,
                size=size,
                label=label,
            )
        )
        changed = True
    work["objects"] = promoted_objects
    return _ensure_canvas_ids(work), changed


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
    st.session_state["exercise_canvas_json"] = loaded_canvas

exercise_data = st.session_state.get("exercise_form_data", {})
canvas_state = _ensure_canvas_ids(st.session_state.get("exercise_canvas_json", _blank_canvas()))
st.session_state["exercise_canvas_json"] = canvas_state

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

control_cols = st.columns([1.1, 1.1, 0.8, 0.8, 2.2], gap="small")
field_mode = control_cols[0].radio("Modo do campo", options=["Adicionar", "Mover"], horizontal=True, key=f"exercise_mode_{selected_exercise_id or 'new'}")
element_type = control_cols[1].selectbox("Elemento", OBJECT_TYPE_OPTIONS, key=f"exercise_element_{selected_exercise_id or 'new'}")
element_color = control_cols[2].color_picker("Cor", value="#ef4444", key=f"exercise_color_{selected_exercise_id or 'new'}")
element_size = control_cols[3].number_input("Tamanho", min_value=8, max_value=48, value=18, step=1, key=f"exercise_size_{selected_exercise_id or 'new'}")
element_label = control_cols[4].text_input("Rotulo opcional", value="", key=f"exercise_label_{selected_exercise_id or 'new'}")

helper_text = "Clica no campo para criar um novo elemento." if field_mode == "Adicionar" else "Clica e arrasta diretamente os elementos no campo."
st.caption(helper_text)

canvas_result = st_canvas(
    fill_color=element_color + "CC",
    stroke_width=3,
    stroke_color=element_color,
    background_image=_build_pitch_background(),
    update_streamlit=True,
    height=CANVAS_HEIGHT,
    width=CANVAS_WIDTH,
    drawing_mode="point" if field_mode == "Adicionar" else "transform",
    initial_drawing=canvas_state,
    display_toolbar=False,
    point_display_radius=2,
    key=f"exercise_canvas_{selected_exercise_id or 'new'}_{field_mode}",
)

raw_canvas = canvas_result.json_data or canvas_state
if field_mode == "Adicionar":
    promoted_canvas, changed = _promote_new_points(
        raw_canvas=raw_canvas,
        element_type=element_type,
        color=element_color,
        size=int(element_size),
        label=_clean_text(element_label),
    )
    st.session_state["exercise_canvas_json"] = promoted_canvas
    if changed:
        st.rerun()
else:
    st.session_state["exercise_canvas_json"] = _ensure_canvas_ids(raw_canvas)

canvas_state = _ensure_canvas_ids(st.session_state["exercise_canvas_json"])
objects_count = len(canvas_state.get("objects", []))
st.caption(f"Elementos no campo: {objects_count}")

action_cols = st.columns([1.1, 1.1, 1.1, 3.7], gap="small")
if action_cols[0].button("Desfazer ultimo", use_container_width=True):
    work = _ensure_canvas_ids(st.session_state["exercise_canvas_json"])
    work["objects"] = work.get("objects", [])[:-1]
    st.session_state["exercise_canvas_json"] = work
    st.rerun()
if action_cols[1].button("Limpar campo", use_container_width=True):
    st.session_state["exercise_canvas_json"] = _blank_canvas()
    st.rerun()
if action_cols[2].button("Gravar exercicio", type="primary", use_container_width=True):
    try:
        if not _clean_text(titulo):
            raise RuntimeError("O titulo do exercicio e obrigatorio.")
        canvas_json = _ensure_canvas_ids(st.session_state["exercise_canvas_json"])
        objects_df = _objects_to_rows(canvas_json)
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
                "canvas_json": json.dumps(canvas_json),
            },
            objects_df,
        )
    except Exception as exc:
        st.error(str(exc))
    else:
        st.session_state["exercise_editor_success"] = f"Exercicio gravado com sucesso. ID: {exercise_id}"
        st.session_state["exercise_loaded_id"] = None
        st.rerun()
if selected_exercise_id and action_cols[3].button("Eliminar exercicio", use_container_width=True):
    delete_exercise(selected_exercise_id)
    st.session_state["exercise_canvas_json"] = _blank_canvas()
    st.session_state["exercise_loaded_id"] = None
    st.session_state["exercise_form_data"] = {}
    st.session_state["exercise_editor_success"] = "Exercicio eliminado com sucesso."
    st.rerun()

success_message = st.session_state.pop("exercise_editor_success", "")
if success_message:
    st.success(success_message)

st.markdown("### Biblioteca de exercicios")
_render_saved_exercises()
