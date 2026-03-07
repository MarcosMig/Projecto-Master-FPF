# app_state.py
# -*- coding: utf-8 -*-

import streamlit as st

DEFAULT_STATE = {
    "auth": False,
    "login_user": "",
    "login_name": "",
    "login_entity": "",
    "process_done": False,
    "df_metrics": None,
    "df_time_audit": None,
    "df_perf": None,
    "df_qc": None,
    "df_samples": None,
    "df_athlete_session": None,
    "report_txt": None,
    "manual_metricas_txt": None,
    "pick_corners": [],
    "pts_gps_picked": None,
    "pick_last_click_sig": None,
}

def init_app_state():
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_processing_outputs():
    st.session_state.process_done = False
    st.session_state.df_metrics = None
    st.session_state.df_time_audit = None
    st.session_state.df_perf = None
    st.session_state.df_qc = None
    st.session_state.df_samples = None
    st.session_state.df_athlete_session = None
    st.session_state.report_txt = None
    st.session_state.manual_metricas_txt = None

def reset_field_picking():
    st.session_state.pick_corners = []
    st.session_state.pts_gps_picked = None
    st.session_state.pick_last_click_sig = None
