# Projecto-Master-FPF

Este repositorio contem o codigo do projeto **FPF Analytics**. Os dados grandes nao estao versionados no Git e devem ser mantidos localmente fora do controlo de versao.

---

## Como Arrancar O Hub

Use sempre o launcher do projeto. No Windows, o comando mais simples e:

```powershell
.\start.bat
```

Tambem pode usar o alias:

```powershell
.\hub.bat
```

Ambos chamam o launcher principal:

```powershell
powershell -ExecutionPolicy Bypass -File .\Start-Hub.ps1
```

Este comando usa um ambiente Python isolado em `%LOCALAPPDATA%\FPF-Hub\.venv`, fora da pasta sincronizada pelo OneDrive. Antes de abrir o Hub, valida dependencias criticas como `streamlit`, `scikit-learn` e `joblib`. Se detetar uma instalacao incompleta, repara o ambiente automaticamente usando `requirements-lock.txt`.

Para forcar uma reinstalacao limpa das dependencias:

```powershell
.\start.bat -Reinstall
```

Evite arrancar com `streamlit run Home.py` diretamente. Esse comando usa o primeiro `streamlit` encontrado no `PATH`, que pode pertencer a outro Python.

---

## Ambiente Manual

Se precisar de configurar um ambiente manualmente:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
```

Use `requirements.txt` apenas para desenvolvimento flexivel. Para o Hub em uso normal, `requirements-lock.txt` mantem as versoes consistentes.

Para correr manualmente com esse ambiente:

```powershell
.\.venv\Scripts\python.exe -m streamlit run Home.py
```

---

## Onde Colocar Os Dados

O codigo espera que os dados existam em:

- `Data/raw_data/` para dados brutos
- `Data/clean_data/` para dados processados

Essas pastas nao sao rastreadas pelo Git.

---

## Criar Pastas Necessarias

Se ainda nao existirem, pode criar as pastas necessarias com:

```powershell
python fpf_modules/constants.py
```

Isto cria `Data/`, `Data/campos/`, `Data/raw_data/` e `Data/clean_data/`.
