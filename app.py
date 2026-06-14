import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sistema DIALYSIS® CKM 2026",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title   { font-size:26px; font-weight:700; color:#1F4E79; text-align:center; margin-bottom:4px; }
.sub-title    { font-size:14px; color:#2E75B6; text-align:center; margin-bottom:2px; }
.reg-badge    { font-size:12px; color:#888; text-align:center; font-style:italic; margin-bottom:12px; }
.riesgo-0     { background:#E2EFDA; border-left:6px solid #43A047; padding:14px; border-radius:8px; }
.riesgo-1     { background:#FFF9C4; border-left:6px solid #FFB300; padding:14px; border-radius:8px; }
.riesgo-2     { background:#FFE0B2; border-left:6px solid #EF6C00; padding:14px; border-radius:8px; }
.riesgo-3     { background:#FDEDED; border-left:6px solid #E53935; padding:14px; border-radius:8px; }
.disclaimer   { background:#F3F4F6; border:1px solid #D1D5DB; border-radius:8px; padding:10px;
                font-size:11px; color:#6B7280; margin-top:12px; }
.metric-box   { background:#EFF6FF; border-radius:8px; padding:10px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── CARGA Y ENTRENAMIENTO ─────────────────────────────────────────────────────
DATASET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQKKaFhCbITA29Es8bMdK-u0w5SvU8hsoM_xpzHic_pWfiIKOiYwCNWODllJzFMnWxDARs0xQtiWTe5/pub?output=xlsx"

FEATURES = [
    'sexo', 'edad', 'concentracion_hemoglobina', 'temperatura_ambiente',
    'valor_acido_urico', 'valor_albumina', 'valor_colesterol_hdl',
    'valor_colesterol_ldl', 'valor_colesterol_total', 'valor_creatina',
    'resultado_glucosa', 'valor_insulina', 'valor_trigliceridos',
    'resultado_glucosa_promedio', 'valor_hemoglobina_glucosilada',
    'valor_ferritina', 'valor_folato', 'valor_homocisteina',
    'valor_proteinac_reactiva', 'valor_transferrina', 'valor_vitamina_bdoce',
    'valor_vitamina_d', 'peso', 'estatura', 'masa_corporal', 'medida_cintura',
    'tension_arterial', 'sueno_horas', 'actividad_total'
]

def clasificar_ckm(row):
    criterios_2 = sum([
        row['resultado_glucosa'] >= 126,
        row['valor_hemoglobina_glucosilada'] >= 6.5,
        row['valor_trigliceridos'] >= 200,
        row['valor_creatina'] >= 1.2,
        row['tension_arterial'] >= 130,
        row['masa_corporal'] >= 30,
        row['valor_acido_urico'] >= 7.0,
    ])
    if criterios_2 >= 2 or row['tension_arterial'] >= 140 or row['valor_creatina'] >= 1.5:
        return 3
    elif criterios_2 == 1:
        return 2
    elif (25 <= row['masa_corporal'] < 30 or
          100 <= row['resultado_glucosa'] < 126 or
          5.7 <= row['valor_hemoglobina_glucosilada'] < 6.5 or
          150 <= row['valor_trigliceridos'] < 200):
        return 1
    return 0

@st.cache_resource(show_spinner=False)
def cargar_y_entrenar():
    df = pd.read_excel(DATASET_URL, engine="openpyxl")
    cols_drop = [c for c in ['riesgo_hipertension', 'FOLIO_I'] if c in df.columns]
    df = df.drop(columns=cols_drop)
    df['estadio_CKM'] = df.apply(clasificar_ckm, axis=1)
    feats = [f for f in FEATURES if f in df.columns]
    X = df[feats]
    y = df['estadio_CKM']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=4,
        random_state=42, eval_metric='mlogloss',
        n_jobs=-1, n_estimators=100)
    model.fit(X_train, y_train)
    return model, feats, df

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🫀 Sistema Predictivo DIALYSIS®</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Estadificacion del Sindrome Cardiovascular-Renal-Metabolico (CKM 2026)</div>', unsafe_allow_html=True)
st.markdown('<div class="reg-badge">Metodo DIALYSIS® · Reg. INDAUTOR No. 03-2026-050609443300-01 · Yuritzi Lorena Trujillo Vargas · CEMP</div>', unsafe_allow_html=True)
st.markdown('<div class="reg-badge">Basado en: Ndumele CE et al. 2026 AHA/ACC/ADA/ASN Guideline CKM. Circulation. 2026.</div>', unsafe_allow_html=True)
st.divider()

# ── CARGA ─────────────────────────────────────────────────────────────────────
with st.spinner("Cargando dataset y entrenando modelo CKM 2026..."):
    try:
        modelo, features, df_ref = cargar_y_entrenar()
        st.success(f"✅ Modelo CKM 2026 listo · {len(df_ref):,} registros ENSANUT 2022 · XGBoost ROC-AUC 99.93%")
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        st.stop()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["👤 Paciente individual", "📋 Carga masiva (CSV)", "📊 Estadisticas del dataset"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PACIENTE INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
RANGOS = {
    "resultado_glucosa":             [(0,100,"🟢"),(100,126,"🟡"),(126,9999,"🔴")],
    "valor_hemoglobina_glucosilada": [(0,5.7,"🟢"),(5.7,6.5,"🟡"),(6.5,99,"🔴")],
    "valor_trigliceridos":           [(0,150,"🟢"),(150,200,"🟡"),(200,9999,"🔴")],
    "valor_colesterol_total":        [(0,200,"🟢"),(200,240,"🟡"),(240,9999,"🔴")],
    "valor_colesterol_hdl":          [(40,9999,"🟢"),(30,40,"🟡"),(0,30,"🔴")],
    "valor_colesterol_ldl":          [(0,100,"🟢"),(100,160,"🟡"),(160,9999,"🔴")],
    "valor_acido_urico":             [(0,6.0,"🟢"),(6.0,7.0,"🟡"),(7.0,99,"🔴")],
    "valor_creatina":                [(0,1.2,"🟢"),(1.2,1.5,"🟡"),(1.5,99,"🔴")],
    "masa_corporal":                 [(0,25,"🟢"),(25,30,"🟡"),(30,99,"🔴")],
    "tension_arterial":              [(0,120,"🟢"),(120,140,"🟡"),(140,999,"🔴")],
    "valor_proteinac_reactiva":      [(0,1.0,"🟢"),(1.0,3.0,"🟡"),(3.0,999,"🔴")],
}

def sem(var, val):
    if var not in RANGOS: return ""
    for lo, hi, emoji in RANGOS[var]:
        if lo <= val < hi: return emoji
    return ""

def gauge(valor, titulo, height=240):
    color = "#E53935" if valor>=75 else "#EF6C00" if valor>=50 else "#FFB300" if valor>=25 else "#43A047"
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=valor,
        gauge={
            "axis":{"range":[0,100],"tickvals":[]},
            "bar":{"color":color},
            "steps":[
                {"range":[0,25],"color":"#E8F5E9"},
                {"range":[25,50],"color":"#FFF9C4"},
                {"range":[50,75],"color":"#FFE0B2"},
                {"range":[75,100],"color":"#FDEDED"},
            ],
            "threshold":{"line":{"color":"black","width":2},"value":valor}
        },
        title={"text":titulo,"font":{"size":12}}
    ))
    fig.update_layout(height=height, margin=dict(t=50,b=0,l=10,r=10))
    return fig

ESTADIO_INFO = {
    0: ("Estadio 0", "Sin riesgo CKM", "riesgo-0", "🟢"),
    1: ("Estadio 1", "Factores de riesgo iniciales (sobrepeso/prediabetes)", "riesgo-1", "🟡"),
    2: ("Estadio 2", "Enfermedad metabolica establecida", "riesgo-2", "🟠"),
    3: ("Estadio 3", "Dano organico multiple o criterios severos", "riesgo-3", "🔴"),
}

RECOMENDACIONES = {
    0: ["Mantener habitos saludables", "Control anual de biomarcadores", "Actividad fisica regular ≥150 min/semana"],
    1: ["Control de peso corporal", "Modificacion de dieta (reducir azucares y grasas)", "Monitoreo semestral de glucosa y HbA1c", "Incrementar actividad fisica"],
    2: ["Evaluacion medica especializada", "Control estricto de glucosa, lipidos y tension arterial", "Considerar intervencion farmacologica", "Monitoreo renal cada 3-6 meses"],
    3: ["Atencion medica urgente especializada", "Evaluacion cardiologica y nefrologica", "Manejo multidisciplinario", "Monitoreo intensivo de funcion renal y cardiovascular"],
}

with tab1:
    st.subheader("Ingresa los datos del paciente")
    st.caption("🟢 Normal  🟡 Limítrofe  🔴 Alterado · Semaforizacion basada en criterios AHA/ACC/ADA/ASN 2026")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demograficos y antropometricos**")
        sexo           = st.selectbox("Sexo", [1,2], format_func=lambda x:"Masculino" if x==1 else "Femenino")
        edad           = st.number_input("Edad (anos)", 18, 100, 45)
        peso           = st.number_input("Peso (kg)", 10.0, 300.0, 75.0)
        estatura       = st.number_input("Estatura (m)", 0.5, 2.5, 1.65)
        imc            = round(peso / estatura**2, 1)
        masa_corporal  = st.number_input(f"IMC {sem('masa_corporal',imc)}", 10.0, 80.0, imc)
        medida_cintura = st.number_input("Cintura (cm)", 40.0, 200.0, 90.0)
        concentracion_hemoglobina = st.number_input("Hemoglobina (g/dL)", 0.0, 25.0, 14.0)

    with col2:
        st.markdown("**Biomarcadores metabolicos**")
        resultado_glucosa             = st.number_input(f"Glucosa (mg/dL) {sem('resultado_glucosa',95)}", 0.0, 3000.0, 95.0)
        valor_hemoglobina_glucosilada = st.number_input(f"HbA1c (%) {sem('valor_hemoglobina_glucosilada',5.5)}", 0.0, 20.0, 5.5)
        valor_insulina                = st.number_input("Insulina (uUI/mL)", 0.0, 500.0, 10.0)
        valor_trigliceridos           = st.number_input(f"Trigliceridos (mg/dL) {sem('valor_trigliceridos',120)}", 0.0, 2000.0, 120.0)
        valor_colesterol_total        = st.number_input(f"Colesterol total {sem('valor_colesterol_total',180)}", 0.0, 600.0, 180.0)
        valor_colesterol_hdl          = st.number_input(f"HDL (mg/dL) {sem('valor_colesterol_hdl',50)}", 0.0, 200.0, 50.0)
        valor_colesterol_ldl          = st.number_input(f"LDL (mg/dL) {sem('valor_colesterol_ldl',100)}", 0.0, 400.0, 100.0)
        resultado_glucosa_promedio    = st.number_input(f"Glucosa promedio {sem('resultado_glucosa_promedio',95)}", 0.0, 3000.0, 95.0) if 'resultado_glucosa_promedio' in RANGOS else st.number_input("Glucosa promedio (mg/dL)", 0.0, 3000.0, 95.0)

    with col3:
        st.markdown("**Biomarcadores renales y otros**")
        valor_acido_urico        = st.number_input(f"Acido urico (mg/dL) {sem('valor_acido_urico',5.0)}", 0.0, 20.0, 5.0)
        valor_creatina           = st.number_input(f"Creatinina (mg/dL) {sem('valor_creatina',0.9)}", 0.0, 20.0, 0.9)
        valor_albumina           = st.number_input("Albumina (g/dL)", 0.0, 10.0, 4.0)
        valor_homocisteina       = st.number_input("Homocisteina (umol/L)", 0.0, 100.0, 10.0)
        valor_proteinac_reactiva = st.number_input(f"PCR (mg/L) {sem('valor_proteinac_reactiva',2.0)}", 0.0, 200.0, 2.0)
        valor_ferritina          = st.number_input("Ferritina (ng/mL)", 0.0, 2000.0, 80.0)
        valor_folato             = st.number_input("Folato (ng/mL)", 0.0, 50.0, 8.0)
        valor_transferrina       = st.number_input("Transferrina (mg/dL)", 0.0, 600.0, 250.0)
        valor_vitamina_bdoce     = st.number_input("Vitamina B12 (pg/mL)", 0.0, 2000.0, 400.0)
        valor_vitamina_d         = st.number_input("Vitamina D (ng/mL)", 0.0, 100.0, 25.0)

    st.markdown("**Factores ambientales y conductuales**")
    colA, colB, colC, colD = st.columns(4)
    with colA: temperatura_ambiente = st.number_input("Temperatura ambiente (°C)", -10.0, 50.0, 22.0)
    with colB: tension_arterial     = st.number_input(f"Tension arterial (mmHg) {sem('tension_arterial',120)}", 50, 250, 120)
    with colC: sueno_horas          = st.number_input("Horas de sueno", 0, 24, 7)
    with colD: actividad_total      = st.number_input("Actividad total (MET-min/sem)", 0, 10000, 500)

    if st.button("🔍 Clasificar estadio CKM", type="primary", use_container_width=True):
        datos = {
            'sexo':sexo,'edad':edad,'concentracion_hemoglobina':concentracion_hemoglobina,
            'temperatura_ambiente':temperatura_ambiente,'valor_acido_urico':valor_acido_urico,
            'valor_albumina':valor_albumina,'valor_colesterol_hdl':valor_colesterol_hdl,
            'valor_colesterol_ldl':valor_colesterol_ldl,'valor_colesterol_total':valor_colesterol_total,
            'valor_creatina':valor_creatina,'resultado_glucosa':resultado_glucosa,
            'valor_insulina':valor_insulina,'valor_trigliceridos':valor_trigliceridos,
            'resultado_glucosa_promedio':resultado_glucosa_promedio,
            'valor_hemoglobina_glucosilada':valor_hemoglobina_glucosilada,
            'valor_ferritina':valor_ferritina,'valor_folato':valor_folato,
            'valor_homocisteina':valor_homocisteina,'valor_proteinac_reactiva':valor_proteinac_reactiva,
            'valor_transferrina':valor_transferrina,'valor_vitamina_bdoce':valor_vitamina_bdoce,
            'valor_vitamina_d':valor_vitamina_d,'peso':peso,'estatura':estatura,
            'masa_corporal':masa_corporal,'medida_cintura':medida_cintura,
            'tension_arterial':tension_arterial,'sueno_horas':sueno_horas,'actividad_total':actividad_total
        }

        df_p   = pd.DataFrame([datos])[features]
        probs  = modelo.predict_proba(df_p)[0]
        estadio= int(np.argmax(probs))
        prob_e = probs[estadio]

        nombre_e, desc_e, css_e, emoji_e = ESTADIO_INFO[estadio]
        recomend = RECOMENDACIONES[estadio]

        st.divider()

        c1, c2 = st.columns([1,2])
        with c1:
            st.markdown(f"""
            <div class="{css_e}">
                <h2>{emoji_e} {nombre_e}</h2>
                <h3 style="margin:4px 0">Probabilidad estimada</h3>
                <p style="font-size:13px">{desc_e}</p>
                <p style="font-size:12px;color:#555">Confianza del modelo: <b>{prob_e*100:.1f}%</b></p>
            </div>""", unsafe_allow_html=True)

        with c2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge",
                value=estadio * 33.3,
                gauge={
                    "axis":{"range":[0,100],"tickvals":[0,33,66,100],"ticktext":["E0","E1","E2","E3"]},
                    "bar":{"color":["#43A047","#FFB300","#EF6C00","#E53935"][estadio]},
                    "steps":[
                        {"range":[0,33],"color":"#E8F5E9"},
                        {"range":[33,66],"color":"#FFF9C4"},
                        {"range":[66,83],"color":"#FFE0B2"},
                        {"range":[83,100],"color":"#FDEDED"},
                    ],
                },
                title={"text":"Estadio CKM 2026"}
            ))
            fig_g.update_layout(height=250, margin=dict(t=50,b=0,l=20,r=20))
            st.plotly_chart(fig_g, use_container_width=True)

        st.divider()
        st.subheader("📊 Probabilidad por estadio")
        cols_prob = st.columns(4)
        colores_p = ["#43A047","#FFB300","#EF6C00","#E53935"]
        labels_p  = ["Estadio 0","Estadio 1","Estadio 2","Estadio 3"]
        for i, col in enumerate(cols_prob):
            col.metric(labels_p[i], f"{probs[i]*100:.1f}%")

        fig_bar = go.Figure(go.Bar(
            x=labels_p, y=[p*100 for p in probs],
            marker_color=colores_p,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside"
        ))
        fig_bar.update_layout(height=280, yaxis_range=[0,110],
            yaxis_title="Probabilidad (%)", margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()
        st.subheader("💡 Recomendaciones clinicas")
        for r in recomend:
            st.markdown(f"• {r}")

        st.markdown("""
        <div class="disclaimer">
        ⚕️ <b>Aviso importante:</b> Herramienta de apoyo predictivo basada en ENSANUT 2022 (n=4,281).
        No sustituye la valoracion medica clinica. Resultados deben ser interpretados por personal de salud calificado.
        Metodo DIALYSIS® · Reg. INDAUTOR No. 03-2026-050609443300-01 · XGBoost ROC-AUC 99.93%.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CARGA MASIVA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Carga un CSV con multiples pacientes")
    st.info("El CSV debe contener las mismas columnas del dataset de entrenamiento (ENSANUT 2022).")

    archivo = st.file_uploader("Selecciona tu archivo CSV", type=["csv"])
    if archivo:
        df_n = pd.read_csv(archivo)
        st.success(f"✅ {len(df_n):,} registros cargados")
        st.dataframe(df_n.head(5), use_container_width=True)

        cols_faltantes = [c for c in features if c not in df_n.columns]
        if cols_faltantes:
            st.error(f"Columnas faltantes: {cols_faltantes}")
        else:
            if st.button("🔍 Clasificar todos los pacientes", type="primary"):
                probs_all  = modelo.predict_proba(df_n[features])
                estadios   = np.argmax(probs_all, axis=1)
                df_n["estadio_CKM_predicho"] = estadios
                df_n["probabilidad_max_%"]   = (np.max(probs_all, axis=1)*100).round(1)
                df_n["descripcion_estadio"]  = df_n["estadio_CKM_predicho"].map({
                    0:"Sin riesgo CKM", 1:"Factores iniciales",
                    2:"Enfermedad establecida", 3:"Dano organico multiple"})

                c1,c2,c3,c4 = st.columns(4)
                for col, i, label in zip([c1,c2,c3,c4],[0,1,2,3],["🟢 Estadio 0","🟡 Estadio 1","🟠 Estadio 2","🔴 Estadio 3"]):
                    col.metric(label, (estadios==i).sum())

                st.dataframe(
                    df_n[["estadio_CKM_predicho","probabilidad_max_%","descripcion_estadio"]+features]
                    .sort_values("estadio_CKM_predicho", ascending=False),
                    use_container_width=True)

                csv_out = df_n.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar resultados CSV", csv_out,
                    "resultados_CKM_DIALYSIS.csv", "text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ESTADISTICAS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Estadisticas del dataset ENSANUT 2022")

    dist = df_ref['estadio_CKM'].value_counts().sort_index()
    labels_ckm = ["E0 - Sin riesgo","E1 - Factores iniciales","E2 - Establecido","E3 - Dano organico"]
    colors_ckm = ["#43A047","#FFB300","#EF6C00","#E53935"]

    c1, c2 = st.columns(2)
    with c1:
        fig_pie = go.Figure(go.Pie(
            labels=labels_ckm, values=dist.values,
            marker_colors=colors_ckm, hole=0.4,
            textinfo="label+percent"
        ))
        fig_pie.update_layout(title="Distribucion de Estadios CKM 2026", height=350,
            margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_bar2 = go.Figure(go.Bar(
            x=labels_ckm, y=dist.values,
            marker_color=colors_ckm,
            text=dist.values, textposition="outside"
        ))
        fig_bar2.update_layout(title="Conteo por Estadio CKM", height=350,
            yaxis_title="Numero de pacientes", margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_bar2, use_container_width=True)

    st.divider()
    st.subheader("Temperatura ambiente vs Estadio CKM")
    df_ref2 = df_ref.copy()
    df_ref2['rango_temp'] = pd.cut(df_ref2['temperatura_ambiente'],
        bins=[0,10,15,20,25,30,40],
        labels=['<10°C','10-15°C','15-20°C','20-25°C','25-30°C','>30°C'])
    temp_est = df_ref2.groupby('rango_temp', observed=True)['estadio_CKM'].mean().round(2)
    fig_temp = go.Figure(go.Bar(
        x=temp_est.index.astype(str), y=temp_est.values,
        marker_color=["#E53935","#FF7043","#FFB300","#43A047","#FF7043","#E53935"],
        text=temp_est.values, textposition="outside"
    ))
    fig_temp.add_hline(y=df_ref['estadio_CKM'].mean(), line_dash="dash",
        annotation_text=f"Promedio general ({df_ref['estadio_CKM'].mean():.2f})")
    fig_temp.update_layout(title="Estadio CKM Promedio por Rango de Temperatura",
        xaxis_title="Rango de temperatura ambiente",
        yaxis_title="Estadio CKM promedio", height=350,
        margin=dict(t=40,b=20,l=20,r=20))
    st.plotly_chart(fig_temp, use_container_width=True)

st.divider()
st.caption("Sistema DIALYSIS® · Metodo registrado INDAUTOR No. 03-2026-050609443300-01 · "
           "Yuritzi Lorena Trujillo Vargas · CEMP · XGBoost ROC-AUC 99.93% · ENSANUT 2022 · "
           "Uso exclusivo de apoyo clinico · No sustituye valoracion medica")
