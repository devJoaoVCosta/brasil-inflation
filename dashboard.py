"""
Dashboard — Evolução dos Índices de Inflação no Brasil (1980–2023)
Indicadores: IPCA, INPC, IPA, IPC-FIPE, INCC, SELIC, Salário Mínimo
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pathlib

# ──────────────────────────────────────────────────────────
# Configuração da página
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inflação Brasil",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# Paleta de cores
# ──────────────────────────────────────────────────────────
BLUE   = "#1591EA"
GRAY   = "#898989"
WHITE  = "#FFFFFF"
DARK   = "#0D1117"
DARK2  = "#161B22"
DARK3  = "#21262D"
BORDER = "#30363D"
BLUE_LIGHT = "#1A7FD4"
BLUE_PALE  = "rgba(21,145,234,0.12)"
TEXT_MUTED = "#8B949E"

PALETTE = [
    "#1591EA", "#F59E0B", "#10B981", "#F43F5E",
    "#A78BFA", "#FB923C", "#34D399", "#60A5FA",
]

# ──────────────────────────────────────────────────────────
# CSS global
# ──────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Fundo geral */
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {DARK};
    }}
    [data-testid="stHeader"] {{ background: transparent; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {DARK2};
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] * {{ color: {WHITE} !important; }}

    /* Métricas */
    [data-testid="metric-container"] {{
        background: {DARK3};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 18px 22px 14px;
        position: relative;
        overflow: hidden;
    }}
    [data-testid="metric-container"]::before {{
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: {BLUE};
        border-radius: 12px 0 0 12px;
    }}
    [data-testid="metric-container"] label {{
        color: {TEXT_MUTED} !important;
        font-size: 0.72rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    [data-testid="stMetricValue"] {{
        color: {WHITE} !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-size: 0.8rem !important;
    }}
    [data-testid="stMetricDelta"] svg {{ display: none; }}

    /* Títulos de seção */
    .section-title {{
        font-size: 0.7rem;
        font-weight: 700;
        color: {BLUE};
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-left: 3px solid {BLUE};
        padding-left: 10px;
        margin-bottom: 12px;
        line-height: 1.2;
    }}

    /* Tabs */
    [data-testid="stTabs"] [role="tab"] {{
        color: {TEXT_MUTED};
        font-size: 0.85rem;
        font-weight: 500;
        border-bottom: 2px solid transparent;
        padding-bottom: 8px;
    }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        color: {BLUE} !important;
        border-bottom: 2px solid {BLUE} !important;
    }}
    [data-testid="stTabsContent"] {{
        padding-top: 16px;
    }}

    /* Selectbox e multiselect */
    [data-testid="stSelectbox"] > div,
    [data-testid="stMultiSelect"] > div {{
        background: {DARK3} !important;
        border-color: {BORDER} !important;
        border-radius: 8px;
        color: {WHITE};
    }}

    /* Slider */
    [data-testid="stSlider"] .stSlider > div {{
        color: {BLUE};
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {DARK2}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}

    /* Divisor */
    hr {{ border-color: {BORDER}; margin: 1.2rem 0; }}

    /* Remove padding extra do main */
    .block-container {{ padding-top: 1.8rem; padding-bottom: 2rem; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# Utilitário de gráfico base
# ──────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    plot_bgcolor=DARK3,
    paper_bgcolor=DARK3,
    font=dict(color=TEXT_MUTED, family="Inter, sans-serif", size=11),
    margin=dict(l=12, r=12, t=28, b=12),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=BORDER,
        font=dict(color=TEXT_MUTED, size=10),
        orientation="h",
        y=-0.18,
    ),
    hoverlabel=dict(bgcolor=DARK2, bordercolor=BORDER, font_color=WHITE),
)

def apply_layout(fig, height=340, **kwargs):
    layout = {**PLOT_LAYOUT, "height": height, **kwargs}
    fig.update_layout(**layout)
    return fig

# ──────────────────────────────────────────────────────────
# Carregamento de dados
# ──────────────────────────────────────────────────────────
_CSV_NAME = "inflacao.csv"
_CANDIDATES = [
    pathlib.Path(__file__).parent / _CSV_NAME,
    pathlib.Path.cwd() / _CSV_NAME,
    pathlib.Path("/mnt/user-data/uploads") / _CSV_NAME,
]

@st.cache_data(show_spinner="Carregando dados...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["data"] = pd.to_datetime(df["referencia"], format="%Y-%m")

    # Normaliza salário mínimo — deflaciona série histórica para comparação relativa
    # Mantém valor nominal mas flag de era
    df["era"] = pd.cut(
        df["ano"],
        bins=[1979, 1993, 2002, 2010, 2016, 2023],
        labels=["Hiperinflação\n(1980–93)", "Pós-Plano Real\n(1994–02)",
                "Boom\n(2003–10)", "Nova Matriz\n(2011–16)", "Recente\n(2017–23)"],
    )

    # Acumulado IPCA desde 1980
    df = df.sort_values("data").reset_index(drop=True)
    df["ipca_acumulado_historico"] = (1 + df["ipca_variacao"] / 100).cumprod() * 100 - 100

    return df

@st.cache_data(show_spinner="Carregando dados...")
def load_data_bytes(uploaded_file) -> pd.DataFrame:
    import io
    df = pd.read_csv(io.BytesIO(uploaded_file.read()))
    df["data"] = pd.to_datetime(df["referencia"], format="%Y-%m")
    df["era"] = pd.cut(
        df["ano"],
        bins=[1979, 1993, 2002, 2010, 2016, 2023],
        labels=["Hiperinflação\n(1980–93)", "Pós-Plano Real\n(1994–02)",
                "Boom\n(2003–10)", "Nova Matriz\n(2011–16)", "Recente\n(2017–23)"],
    )
    df = df.sort_values("data").reset_index(drop=True)
    df["ipca_acumulado_historico"] = (1 + df["ipca_variacao"] / 100).cumprod() * 100 - 100
    return df

# Resolve caminho
_csv_path = next((str(p) for p in _CANDIDATES if p.exists()), None)

if _csv_path is None:
    st.markdown(
        f"<h2 style='color:{WHITE};'>🇧🇷 Inflação Brasil</h2>",
        unsafe_allow_html=True,
    )
    st.info("**Dataset não encontrado.** Faça o upload do arquivo `inflacao.csv` abaixo.", icon="📂")
    uploaded = st.file_uploader("Selecione inflacao.csv", type=["csv"])
    if uploaded is None:
        st.stop()
    df_full = load_data_bytes(uploaded)
else:
    df_full = load_data(_csv_path)

# ──────────────────────────────────────────────────────────
# Sidebar — Filtros
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>"
        f"<span style='font-size:28px;'>🇧🇷</span>"
        f"<div><p style='margin:0;font-size:1rem;font-weight:700;color:{WHITE};'>Inflação Brasil</p>"
        f"<p style='margin:0;font-size:0.75rem;color:{TEXT_MUTED};'>1980 – 2023</p></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(f"<p style='font-size:0.7rem;font-weight:700;color:{BLUE};text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;'>Período</p>", unsafe_allow_html=True)
    ano_min, ano_max = int(df_full["ano"].min()), int(df_full["ano"].max())
    ano_range = st.slider("", ano_min, ano_max, (ano_min, ano_max), label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"<p style='font-size:0.7rem;font-weight:700;color:{BLUE};text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;'>Indicadores</p>", unsafe_allow_html=True)

    indicadores_map = {
        "IPCA": "ipca_variacao",
        "INPC": "inpc_variacao",
        "IPA":  "ipa_variacao",
        "IPC-FIPE": "ipc_fipe_variacao",
        "INCC": "incc_variacao",
    }
    indicadores_sel = st.multiselect(
        "Índices",
        list(indicadores_map.keys()),
        default=["IPCA", "INPC"],
        label_visibility="collapsed",
    )
    if not indicadores_sel:
        indicadores_sel = ["IPCA"]

    st.markdown("---")
    st.markdown(f"<p style='font-size:0.7rem;font-weight:700;color:{BLUE};text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;'>Visão temporal</p>", unsafe_allow_html=True)
    granularidade = st.radio(
        "",
        ["Mensal", "Anual"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"<p style='font-size:0.7rem;color:{TEXT_MUTED};line-height:1.5;'>"
        f"Fonte: <a href='https://www.kaggle.com/datasets/fidelissauro/inflacao-brasil' "
        f"style='color:{BLUE};' target='_blank'>Kaggle</a> · IBGE / BACEN</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────
# Filtragem
# ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def filter_df(df, ano_ini, ano_fim):
    return df[(df["ano"] >= ano_ini) & (df["ano"] <= ano_fim)].copy()

df = filter_df(df_full, ano_range[0], ano_range[1])

# ──────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(
        f"<h1 style='color:{WHITE};font-size:1.85rem;font-weight:800;margin:0;'>"
        f"🇧🇷 Evolução dos Índices de Inflação no Brasil</h1>"
        f"<p style='color:{TEXT_MUTED};font-size:0.88rem;margin-top:4px;margin-bottom:0;'>"
        f"IPCA · INPC · IPA · IPC-FIPE · INCC · SELIC · Salário Mínimo &nbsp;·&nbsp; "
        f"{ano_range[0]} – {ano_range[1]}</p>",
        unsafe_allow_html=True,
    )
with col_h2:
    registros = len(df)
    st.markdown(
        f"<div style='text-align:right;padding-top:8px;'>"
        f"<span style='font-size:0.72rem;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:.08em;'>Registros</span><br>"
        f"<span style='font-size:1.6rem;font-weight:700;color:{BLUE};'>{registros:,}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ──────────────────────────────────────────────────────────
# KPIs
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Indicadores do Período Selecionado</div>', unsafe_allow_html=True)

ipca_med   = df["ipca_variacao"].mean()
inpc_med   = df["inpc_variacao"].mean()
ipca_max   = df["ipca_variacao"].max()
ipca_max_ref = df.loc[df["ipca_variacao"].idxmax(), "referencia"]
selic_med  = df[df["selic_meta"] > 0]["selic_meta"].mean() if (df["selic_meta"] > 0).any() else 0
salario_atual = df.sort_values("data").iloc[-1]["salario_minimo"]
ipca_acum  = df["ipca_acumulado_ano"].iloc[-1] if len(df) else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("IPCA Médio Mensal",  f"{ipca_med:.2f}%",  f"INPC: {inpc_med:.2f}%")
k2.metric("Pico do IPCA",       f"{ipca_max:.2f}%",  f"em {ipca_max_ref}")
k3.metric("SELIC Média",        f"{selic_med:.1f}%" if selic_med else "—", "Meta ao ano")
k4.metric("Último Sal. Mínimo", f"R$ {salario_atual:,.0f}".replace(",", "."), f"em {df.sort_values('data').iloc[-1]['referencia']}")
k5.metric("IPCA Acum. (últ. ano)", f"{ipca_acum:.2f}%", "acumulado 12m")

st.markdown("---")

# ──────────────────────────────────────────────────────────
# TABS principais
# ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Evolução Temporal",
    "📊  Comparação de Índices",
    "💰  SELIC & Juros Reais",
    "👷  Salário Mínimo",
])

# ═══════════════════════════════════
# TAB 1 — EVOLUÇÃO TEMPORAL
# ═══════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="section-title">Variação Mensal dos Índices Selecionados</div>', unsafe_allow_html=True)

        if granularidade == "Mensal":
            df_plot = df.copy()
            x_col = "data"
        else:
            agg = {indicadores_map[i]: "sum" for i in indicadores_sel}
            df_plot = df.groupby("ano").agg(agg).reset_index()
            df_plot["data"] = pd.to_datetime(df_plot["ano"].astype(str) + "-01-01")
            x_col = "data"

        fig_line = go.Figure()
        for i, nome in enumerate(indicadores_sel):
            col_name = indicadores_map[nome]
            fig_line.add_trace(go.Scatter(
                x=df_plot[x_col], y=df_plot[col_name],
                name=nome,
                mode="lines",
                line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
                fill="tozeroy" if nome == "IPCA" else "none",
                fillcolor=BLUE_PALE if nome == "IPCA" else "rgba(0,0,0,0)",
                hovertemplate=f"<b>{nome}</b><br>%{{x|%b %Y}}: %{{y:.2f}}%<extra></extra>",
            ))

        # Linhas de eventos históricos
        eventos = {
            "1994-07": "Plano Real",
            "2002-10": "Crise Lula",
            "2008-09": "Crise Glob.",
            "2015-01": "Recessão",
            "2020-03": "COVID-19",
        }
        for data_ev, label in eventos.items():
            ano_ev = int(data_ev[:4])
            if ano_range[0] <= ano_ev <= ano_range[1]:
                fig_line.add_vline(
                    x=pd.Timestamp(data_ev).timestamp() * 1000,
                    line_dash="dot", line_color=GRAY, line_width=1,
                    annotation_text=label,
                    annotation_font=dict(color=GRAY, size=9),
                    annotation_position="top right",
                )

        apply_layout(fig_line, height=360)
        fig_line.update_layout(
            yaxis_title="Variação (%)",
            xaxis_title="",
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Top 10 maiores IPCAs</div>', unsafe_allow_html=True)
        top10 = df.nlargest(10, "ipca_variacao")[["referencia", "ipca_variacao"]].reset_index(drop=True)
        top10.columns = ["Mês", "IPCA (%)"]
        top10["IPCA (%)"] = top10["IPCA (%)"].round(2)

        fig_top = go.Figure(go.Bar(
            y=top10["Mês"][::-1],
            x=top10["IPCA (%)"][::-1],
            orientation="h",
            marker=dict(
                color=top10["IPCA (%)"][::-1],
                colorscale=[[0, BLUE], [1, "#F43F5E"]],
                showscale=False,
            ),
            text=[f"{v:.1f}%" for v in top10["IPCA (%)"][::-1]],
            textposition="outside",
            textfont=dict(color=WHITE, size=10),
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        ))
        apply_layout(fig_top, height=360)
        fig_top.update_layout(
            xaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT_MUTED, size=9)),
            margin=dict(l=8, r=40, t=28, b=12),
        )
        st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")

    # Heatmap mensal
    st.markdown('<div class="section-title">Heatmap IPCA — Mês × Ano</div>', unsafe_allow_html=True)
    df_heat = df.pivot_table(values="ipca_variacao", index="mes", columns="ano")
    meses_label = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
    df_heat.index = [meses_label[m-1] for m in df_heat.index]

    fig_heat = go.Figure(go.Heatmap(
        z=df_heat.values,
        x=[str(c) for c in df_heat.columns],
        y=df_heat.index.tolist(),
        colorscale=[[0, DARK3], [0.3, BLUE], [0.7, "#F59E0B"], [1, "#F43F5E"]],
        text=np.round(df_heat.values, 1),
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorbar=dict(
            title=dict(text="IPCA %", font=dict(color=TEXT_MUTED)),
            tickfont=dict(color=TEXT_MUTED),
            thickness=12,
        ),
        hovertemplate="<b>%{y} / %{x}</b><br>IPCA: %{z:.2f}%<extra></extra>",
    ))
    apply_layout(fig_heat, height=300)
    fig_heat.update_layout(
        xaxis=dict(tickfont=dict(color=TEXT_MUTED, size=9), side="bottom"),
        yaxis=dict(tickfont=dict(color=TEXT_MUTED, size=10), autorange="reversed"),
        margin=dict(l=40, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════
# TAB 2 — COMPARAÇÃO DE ÍNDICES
# ═══════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Acumulado Anual — todos os índices</div>', unsafe_allow_html=True)
        acum_cols = {
            "IPCA": "ipca_acumulado_ano",
            "INPC": "inpc_acumulado_ano",
            "IPA":  "ipa_acumulado_ano",
            "IPC-FIPE": "ipc_fipe_acumulado_ano",
            "INCC": "incc_acumulado_ano",
        }
        df_anual = df[df["consolidado_ano"]].copy()
        if df_anual.empty:
            df_anual = df.groupby("ano").last().reset_index()

        fig_acum = go.Figure()
        for i, (nome, col) in enumerate(acum_cols.items()):
            fig_acum.add_trace(go.Bar(
                name=nome,
                x=df_anual["ano"],
                y=df_anual[col],
                marker_color=PALETTE[i % len(PALETTE)],
                hovertemplate=f"<b>{nome}</b> %{{x}}: %{{y:.2f}}%<extra></extra>",
            ))
        fig_acum.update_layout(barmode="group")
        apply_layout(fig_acum, height=350)
        fig_acum.update_layout(
            yaxis_title="Acumulado (%)",
            legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
        )
        st.plotly_chart(fig_acum, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">IPCA vs INPC — Diferença mensal</div>', unsafe_allow_html=True)
        df_diff = df.copy()
        df_diff["diff"] = df_diff["ipca_variacao"] - df_diff["inpc_variacao"]

        fig_diff = go.Figure()
        fig_diff.add_trace(go.Bar(
            x=df_diff["data"],
            y=df_diff["diff"],
            marker=dict(
                color=df_diff["diff"].apply(lambda v: BLUE if v >= 0 else "#F43F5E"),
            ),
            hovertemplate="<b>%{x|%b %Y}</b><br>Diferença: %{y:.2f}pp<extra></extra>",
            name="IPCA − INPC",
        ))
        fig_diff.add_hline(y=0, line_color=GRAY, line_width=1)
        apply_layout(fig_diff, height=350)
        fig_diff.update_layout(
            yaxis_title="Diferença (pp)",
            annotations=[dict(
                x=0.01, y=0.97, xref="paper", yref="paper",
                text="Azul = IPCA > INPC  |  Vermelho = INPC > IPCA",
                showarrow=False, font=dict(color=GRAY, size=9),
                align="left",
            )],
        )
        st.plotly_chart(fig_diff, use_container_width=True)

    st.markdown("---")

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Boxplot por Era Econômica — IPCA mensal</div>', unsafe_allow_html=True)
        df_era = df.dropna(subset=["era"]).copy()
        eras = df_era["era"].cat.categories.tolist()
        fig_box = go.Figure()
        for i, era in enumerate(eras):
            sub = df_era[df_era["era"] == era]["ipca_variacao"]
            fig_box.add_trace(go.Box(
                y=sub,
                name=era.replace("\n", " "),
                marker_color=PALETTE[i % len(PALETTE)],
                line_color=PALETTE[i % len(PALETTE)],
                fillcolor=f"rgba({int(PALETTE[i%len(PALETTE)][1:3],16)},"
                          f"{int(PALETTE[i%len(PALETTE)][3:5],16)},"
                          f"{int(PALETTE[i%len(PALETTE)][5:7],16)},0.2)",
                boxmean=True,
            ))
        apply_layout(fig_box, height=340)
        fig_box.update_layout(
            showlegend=False,
            yaxis_title="IPCA mensal (%)",
            xaxis=dict(tickfont=dict(size=9, color=TEXT_MUTED)),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">Scatter IPCA × INPC (por ano)</div>', unsafe_allow_html=True)
        df_sc = df.groupby("ano")[["ipca_variacao","inpc_variacao"]].mean().reset_index()
        fig_sc = px.scatter(
            df_sc, x="ipca_variacao", y="inpc_variacao",
            text="ano",
            color="ano",
            color_continuous_scale=[[0, DARK3], [0.5, BLUE], [1, "#F59E0B"]],
            labels={"ipca_variacao":"IPCA médio (%)", "inpc_variacao":"INPC médio (%)"},
            template="plotly_dark",
        )
        fig_sc.update_traces(
            textposition="top center",
            textfont=dict(size=8, color=TEXT_MUTED),
            marker=dict(size=8, opacity=0.85),
        )
        # Linha de identidade
        lim = max(df_sc["ipca_variacao"].max(), df_sc["inpc_variacao"].max()) * 1.05
        fig_sc.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim],
            mode="lines",
            line=dict(color=GRAY, width=1, dash="dot"),
            name="IPCA = INPC",
            showlegend=False,
        ))
        apply_layout(fig_sc, height=340)
        fig_sc.update_layout(
            coloraxis_showscale=False,
            showlegend=False,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ═══════════════════════════════════
# TAB 3 — SELIC & JUROS REAIS
# ═══════════════════════════════════
with tab3:
    df_selic = df[df["selic_meta"] > 0].copy()

    col_s1, col_s2 = st.columns([3, 2])

    with col_s1:
        st.markdown('<div class="section-title">SELIC Meta vs IPCA Acumulado 12 meses</div>', unsafe_allow_html=True)
        fig_selic = make_subplots(specs=[[{"secondary_y": True}]])
        fig_selic.add_trace(
            go.Scatter(
                x=df_selic["data"], y=df_selic["selic_meta"],
                name="SELIC Meta (%)",
                line=dict(color=BLUE, width=2.5),
                fill="tozeroy", fillcolor=BLUE_PALE,
                hovertemplate="SELIC: %{y:.1f}%<extra></extra>",
            ), secondary_y=False,
        )
        fig_selic.add_trace(
            go.Scatter(
                x=df["data"], y=df["ipca_acumulado_doze_meses"],
                name="IPCA 12m (%)",
                line=dict(color="#F59E0B", width=2, dash="dot"),
                hovertemplate="IPCA 12m: %{y:.1f}%<extra></extra>",
            ), secondary_y=True,
        )
        apply_layout(fig_selic, height=360)
        fig_selic.update_layout(
            legend=dict(orientation="h", y=-0.18),
            plot_bgcolor=DARK3, paper_bgcolor=DARK3,
        )
        fig_selic.update_yaxes(
            title_text="SELIC (%)", secondary_y=False,
            gridcolor=BORDER, tickfont=dict(color=TEXT_MUTED),
        )
        fig_selic.update_yaxes(
            title_text="IPCA 12m (%)", secondary_y=True,
            tickfont=dict(color="#F59E0B"),
        )
        st.plotly_chart(fig_selic, use_container_width=True)

    with col_s2:
        st.markdown('<div class="section-title">Juros Reais ex-post (%)</div>', unsafe_allow_html=True)
        df_jr = df[(df["juros_reais"] != 0) & (df["juros_reais"].abs() < 500)].copy()
        fig_jr = go.Figure()
        fig_jr.add_trace(go.Scatter(
            x=df_jr["data"], y=df_jr["juros_reais"],
            mode="lines",
            line=dict(color="#10B981", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(16,185,129,0.1)",
            hovertemplate="%{x|%b %Y}<br>Juro real: %{y:.2f}%<extra></extra>",
        ))
        fig_jr.add_hline(y=0, line_color=GRAY, line_width=1)
        apply_layout(fig_jr, height=360)
        fig_jr.update_layout(yaxis_title="Juros Reais (%)")
        st.plotly_chart(fig_jr, use_container_width=True)

    st.markdown("---")

    # SELIC por ano
    st.markdown('<div class="section-title">SELIC Meta anual — evolução histórica</div>', unsafe_allow_html=True)
    df_selic_ano = df_selic.groupby("ano")["selic_meta"].mean().reset_index()
    fig_selic_bar = go.Figure(go.Bar(
        x=df_selic_ano["ano"],
        y=df_selic_ano["selic_meta"],
        marker=dict(
            color=df_selic_ano["selic_meta"],
            colorscale=[[0, BLUE], [0.5, "#F59E0B"], [1, "#F43F5E"]],
            showscale=False,
        ),
        text=df_selic_ano["selic_meta"].round(1).astype(str) + "%",
        textposition="outside",
        textfont=dict(size=8, color=WHITE),
        hovertemplate="<b>%{x}</b><br>SELIC: %{y:.1f}%<extra></extra>",
    ))
    apply_layout(fig_selic_bar, height=280)
    fig_selic_bar.update_layout(yaxis_title="SELIC média (%)")
    st.plotly_chart(fig_selic_bar, use_container_width=True)

# ═══════════════════════════════════
# TAB 4 — SALÁRIO MÍNIMO
# ═══════════════════════════════════
with tab4:
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown('<div class="section-title">Evolução do Salário Mínimo (nominal)</div>', unsafe_allow_html=True)
        df_sal = df.groupby("ano")["salario_minimo"].last().reset_index()
        fig_sal = go.Figure()
        fig_sal.add_trace(go.Scatter(
            x=df_sal["ano"], y=df_sal["salario_minimo"],
            mode="lines+markers",
            line=dict(color=BLUE, width=2.5),
            marker=dict(color=BLUE, size=5),
            fill="tozeroy", fillcolor=BLUE_PALE,
            hovertemplate="<b>%{x}</b><br>R$ %{y:,.2f}<extra></extra>",
        ))
        apply_layout(fig_sal, height=340)
        fig_sal.update_layout(
            yaxis_title="Salário Mínimo (R$)",
            yaxis_type="log",
        )
        fig_sal.add_annotation(
            x=0.02, y=0.95, xref="paper", yref="paper",
            text="Escala logarítmica", showarrow=False,
            font=dict(color=GRAY, size=9),
        )
        st.plotly_chart(fig_sal, use_container_width=True)

    with col_m2:
        st.markdown('<div class="section-title">Variação real do salário mínimo vs IPCA (anual)</div>', unsafe_allow_html=True)
        df_sal_real = df[df["consolidado_ano"]].groupby("ano").agg(
            salario=("salario_minimo", "last"),
            ipca_acum=("ipca_acumulado_ano", "last"),
        ).reset_index()
        if df_sal_real.empty:
            df_sal_real = df.groupby("ano").agg(
                salario=("salario_minimo", "last"),
                ipca_acum=("ipca_acumulado_ano", "last"),
            ).reset_index()
        df_sal_real["var_salario"] = df_sal_real["salario"].pct_change() * 100
        df_sal_real["ganho_real"] = df_sal_real["var_salario"] - df_sal_real["ipca_acum"]
        df_sal_real = df_sal_real.dropna()

        fig_ganho = go.Figure()
        fig_ganho.add_trace(go.Bar(
            x=df_sal_real["ano"],
            y=df_sal_real["var_salario"],
            name="Variação nominal (%)",
            marker_color=BLUE,
            opacity=0.8,
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig_ganho.add_trace(go.Scatter(
            x=df_sal_real["ano"],
            y=df_sal_real["ipca_acum"],
            name="IPCA acumulado (%)",
            line=dict(color="#F43F5E", width=2, dash="dot"),
            hovertemplate="%{x} IPCA: %{y:.1f}%<extra></extra>",
        ))
        apply_layout(fig_ganho, height=340)
        fig_ganho.update_layout(
            barmode="overlay",
            yaxis_title="(%)",
            legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
        )
        st.plotly_chart(fig_ganho, use_container_width=True)

    st.markdown("---")

    # Ganho/perda real acumulado
    st.markdown('<div class="section-title">Poder de compra acumulado — Salário Mínimo deflacionado pelo IPCA</div>', unsafe_allow_html=True)
    df_poder = df.copy()
    df_poder["ipca_idx"] = (1 + df_poder["ipca_variacao"] / 100).cumprod()
    df_poder["salario_real"] = df_poder["salario_minimo"] / df_poder["ipca_idx"]
    df_poder_anual = df_poder.groupby("ano").last().reset_index()

    fig_poder = make_subplots(specs=[[{"secondary_y": True}]])
    fig_poder.add_trace(
        go.Scatter(
            x=df_poder_anual["ano"], y=df_poder_anual["salario_real"],
            name="Salário real (deflacionado)",
            line=dict(color="#10B981", width=2.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
        ), secondary_y=False,
    )
    fig_poder.add_trace(
        go.Scatter(
            x=df_poder_anual["ano"], y=df_poder_anual["salario_minimo"],
            name="Salário nominal",
            line=dict(color=BLUE, width=1.5, dash="dot"),
        ), secondary_y=True,
    )
    apply_layout(fig_poder, height=300)
    fig_poder.update_layout(
        legend=dict(orientation="h", y=-0.2),
        plot_bgcolor=DARK3, paper_bgcolor=DARK3,
    )
    fig_poder.update_yaxes(
        title_text="Real (base 1980)", secondary_y=False,
        gridcolor=BORDER, tickfont=dict(color=TEXT_MUTED),
    )
    fig_poder.update_yaxes(
        title_text="Nominal (R$)", secondary_y=True,
        type="log", tickfont=dict(color=BLUE),
    )
    st.plotly_chart(fig_poder, use_container_width=True)

# ──────────────────────────────────────────────────────────
# Tabela resumo
# ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">Tabela Resumo Anual</div>', unsafe_allow_html=True)

df_tabela = df.groupby("ano").agg(
    IPCA_Acum   =("ipca_acumulado_ano",   "last"),
    INPC_Acum   =("inpc_acumulado_ano",   "last"),
    IPA_Acum    =("ipa_acumulado_ano",    "last"),
    SELIC       =("selic_meta",           "last"),
    Salario_Min =("salario_minimo",       "last"),
).reset_index()
df_tabela.columns = ["Ano", "IPCA (%)", "INPC (%)", "IPA (%)", "SELIC (%)", "Sal. Mínimo (R$)"]
df_tabela = df_tabela.sort_values("Ano", ascending=False)

st.dataframe(
    df_tabela.style
    .background_gradient(subset=["IPCA (%)"], cmap="RdYlGn_r")
    .background_gradient(subset=["SELIC (%)"], cmap="Blues")
    .format({
        "IPCA (%)": "{:.2f}",
        "INPC (%)": "{:.2f}",
        "IPA (%)":  "{:.2f}",
        "SELIC (%)":"{:.1f}",
        "Sal. Mínimo (R$)": "R$ {:,.2f}",
    }),
    use_container_width=True,
    height=420,
)

st.caption(
    f"Fonte: IBGE, BACEN · Dataset: Kaggle (fidelissauro/inflacao-brasil) · "
    f"Período: {ano_range[0]}–{ano_range[1]} · {registros} registros mensais"
)
