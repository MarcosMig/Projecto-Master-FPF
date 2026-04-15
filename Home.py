import streamlit as st


st.set_page_config(page_title="FPF Analytics Hub", layout="wide")


st.markdown(
	"""
	<style>
	.info-floating-button {
		position: fixed;
		right: 1.5rem;
		bottom: 1.5rem;
		z-index: 9999;
	}

	.version-floating-label {
		position: fixed;
		left: calc(15rem + 4.5rem);
		bottom: 1.5rem;
		z-index: 9999;
		padding: 0.55rem 0.8rem;
		border-radius: 999px;
		border: 1px solid rgba(255, 255, 255, 0.14);
		background: rgba(28, 28, 36, 0.9);
		color: #d8d8df;
		font-size: 0.88rem;
		line-height: 1;
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
		backdrop-filter: blur(6px);
	}

	@media (max-width: 900px) {
		.version-floating-label {
			left: 1rem;
			bottom: 4.5rem;
		}
	}

	.info-floating-button .info-icon {
		width: 2.5rem;
		height: 2.5rem;
		border-radius: 999px;
		border: 1px solid rgba(255, 255, 255, 0.18);
		background: rgba(28, 28, 36, 0.92);
		color: #f3f3f3;
		display: flex;
		align-items: center;
		justify-content: center;
		font-weight: 700;
		font-size: 1rem;
		cursor: default;
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
		backdrop-filter: blur(6px);
	}

	.info-floating-button .info-tooltip {
		position: absolute;
		right: 0;
		bottom: 3.25rem;
		min-width: 220px;
		padding: 0.8rem 0.95rem;
		border-radius: 0.75rem;
		background: rgba(20, 20, 27, 0.96);
		border: 1px solid rgba(255, 255, 255, 0.14);
		color: #f3f3f3;
		box-shadow: 0 12px 32px rgba(0, 0, 0, 0.32);
		opacity: 0;
		visibility: hidden;
		transform: translateY(8px);
		transition: opacity 0.18s ease, transform 0.18s ease, visibility 0.18s ease;
	}

	.info-floating-button:hover .info-tooltip {
		opacity: 1;
		visibility: visible;
		transform: translateY(0);
	}

	.info-tooltip-title {
		font-size: 0.8rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: #a9a9b3;
		margin-bottom: 0.45rem;
	}

	.info-tooltip-name {
		font-size: 0.95rem;
		line-height: 1.45;
		margin: 0;
	}

	.home-subtitle {
		font-size: 1.2rem;
		line-height: 1.6;
		color: #d8d8df;
		margin-bottom: 1.5rem;
		max-width: 760px;
	}

	a.nav-card-link {
		display: block;
		text-decoration: none;
		color: inherit;
	}

	.nav-card {
		padding: 1rem 1.05rem;
		border: 1px solid rgba(255, 255, 255, 0.14);
		border-radius: 0.9rem;
		background: rgba(255, 255, 255, 0.03);
		min-height: 145px;
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		gap: 0.7rem;
		transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
	}

	a.nav-card-link:hover .nav-card {
		transform: translateY(-2px);
		border-color: rgba(255, 255, 255, 0.28);
		background: rgba(255, 255, 255, 0.05);
		box-shadow: 0 14px 34px rgba(0, 0, 0, 0.2);
	}

	.nav-card-title {
		font-size: 1.2rem;
		font-weight: 600;
		margin-bottom: 0.35rem;
		color: #f5f5f7;
	}

	.nav-card-text {
		font-size: 0.96rem;
		line-height: 1.5;
		color: #d4d4db;
		margin-bottom: 0;
	}

	.nav-card-cta {
		font-size: 0.92rem;
		font-weight: 600;
		color: #f5f5f7;
	}

	.nav-card-body {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
	}
	</style>
	""",
	unsafe_allow_html=True,
)


st.header("FPF Analytics Hub")
st.markdown(
	'<div class="home-subtitle">Plataforma para análise de performance e gestão de dados da FPF.</div>',
	unsafe_allow_html=True,
)

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
	st.markdown(
		"""
		<a class="nav-card-link" href="/Inserc%C3%A3o_Dados" target="_self">
			<div class="nav-card">
				<div class="nav-card-body">
					<div class="nav-card-title">📥 Insercão Dados</div>
					<div class="nav-card-text">Carrega novas sessões, atualiza os datasets e prepara os dados para análise.</div>
				</div>
				<div class="nav-card-cta">Abrir página</div>
			</div>
		</a>
		""",
		unsafe_allow_html=True,
	)

with nav_col2:
	st.markdown(
		"""
		<a class="nav-card-link" href="/An%C3%A1lise_Performance" target="_self">
			<div class="nav-card">
				<div class="nav-card-body">
					<div class="nav-card-title">📊 Análise Performance</div>
					<div class="nav-card-text">Consulta métricas, visualizações posicionais e indicadores de qualidade do tracking.</div>
				</div>
				<div class="nav-card-cta">Abrir página</div>
			</div>
		</a>
		""",
		unsafe_allow_html=True,
	)

st.markdown(
	"""
	<div class="version-floating-label">Versão 1.0</div>

	<div class="info-floating-button" aria-label="Informação do projeto">
		<div class="info-icon">i</div>
		<div class="info-tooltip">
			<div class="info-tooltip-title">Desenvolvimento</div>
			<p class="info-tooltip-name">Marcos Cardoso</p>
			<p class="info-tooltip-name">Belo Matos</p>
			<p class="info-tooltip-name">João Marques</p>
		</div>
	</div>
	""",
	unsafe_allow_html=True,
)
