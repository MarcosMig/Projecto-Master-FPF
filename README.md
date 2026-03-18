# Projecto-Master-FPF

Este repositório contém o código do projeto **FPF Analytics**. Os dados (arquivos grandes) **não estão versionados** no Git e devem ser mantidos localmente fora do controle de versão.

---

## 🚀 Como configurar o ambiente (local)

1. Crie/ative um venv (recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instale dependências:

```powershell
pip install -r requirements.txt
```

---

## 📁 Onde colocar os dados (não versionados)

O código espera que os dados existam em:

- `Data/raw_data/` (dados brutos)
- `Data/clean_data/` (dados processados)

Essas pastas **não são rastreadas pelo Git** (elas estão em `.gitignore`).

> ✅ Se precisar, crie seus dados nessa pasta manualmente ou escreva um script para baixá-los / gerá-los.

---

## 🛠️ Criar as pastas necessárias (se ainda não existirem)

Você pode criar as pastas necessárias executando:

```powershell
python fpf_modules/constants.py
```

Isso irá criar `Data/`, `Data/campos/`, `Data/raw_data/` e `Data/clean_data/` se ainda não existirem.

---

## 📌 Nota sobre o histórico do Git

Os arquivos grandes que estavam em `Data/clean_data/` foram removidos do histórico para que este repositório possa ser enviado ao GitHub sem atingir os limites de tamanho.

Se você clonar este repositório em outra máquina, basta colocar seus arquivos grandes em `Data/clean_data/` localmente, e eles não serão incluídos no Git.

---

## 🧪 Como rodar

Dependendo do que você quiser testar, execute:

```powershell
streamlit run Home.py
```

ou qualquer outro script que você esteja usando como ponto de entrada.
