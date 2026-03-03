import hashlib
import re

import pandas as pd


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace('"', "") for c in df.columns]
    return df


def get_atleta_id(fname: str) -> str:
    m = re.search(r"Player-(\d+)", fname, flags=re.I)
    return m.group(1) if m else (fname.split("-")[0] if "-" in fname else fname)


def infer_fase(fname: str) -> str:
    n = fname.upper()
    if "WARM" in n or "WUP" in n:
        return "Warm-Up"
    if "1P" in n or "PRIMEIRA" in n or "FIRST" in n:
        return "1P"
    if "2P" in n or "SEGUNDA" in n or "SECOND" in n:
        return "2P"
    return "Extra"


def read_csv_upload(upload, nrows=None) -> pd.DataFrame:
    try:
        upload.seek(0)
    except Exception:
        pass
    df = pd.read_csv(upload, sep=None, engine="python", nrows=nrows)
    try:
        upload.seek(0)
    except Exception:
        pass
    return clean_cols(df)


def hash_session(data_sessao, selecao, genero, contexto, estadio, f_campo_files, f_atleta_files) -> str:
    h = hashlib.sha1()
    h.update(str(data_sessao).encode("utf-8"))
    h.update(str(selecao).encode("utf-8"))
    h.update(str(genero).encode("utf-8"))
    h.update(str(contexto).encode("utf-8"))
    h.update(str(estadio).encode("utf-8"))

    def _feed_files(files):
        for uf in sorted(files, key=lambda x: x.name):
            h.update(uf.name.encode("utf-8"))
            try:
                h.update(str(getattr(uf, "size", "")).encode("utf-8"))
            except Exception:
                pass
            try:
                uf.seek(0)
                chunk = uf.read(8192)
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", errors="ignore")
                h.update(chunk or b"")
                uf.seek(0)
            except Exception:
                pass

    _feed_files(f_campo_files)
    _feed_files(f_atleta_files)
    return h.hexdigest()
