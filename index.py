import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.visualization.wcsaxes import SphericalCircle, Quadrangle

from astropy.coordinates import SkyCoord, ICRS, FK5
from astropy import units as u
import ligo.skymap.plot

import os

data_dir = 'MKW4/'

def custom_axes(a):
    a.minorticks_on()
    a.yaxis.set_ticks_position ( 'both' )
    a.xaxis.set_ticks_position ( 'both' )
    a.tick_params ( which = 'major', direction = 'inout', length = 10 )
    a.tick_params ( which = 'minor', direction = 'in', length = 5 )
    a.tick_params ( direction = 'in', pad = 10 )  
    a.tick_params ( which = 'both', width = 2 )
    # La siguiente línea obliga al grafico a ser cuadrado:
    # a.set_aspect ( 1.0/a.get_data_ratio(), adjustable = 'box' )
    a.grid ( which = 'major', color = 'black', linestyle = '--', linewidth = '1.0', alpha = 0.2, zorder = -1 )
    a.grid ( which = 'minor', color = 'gray', linestyle = '-', linewidth = '1.0', alpha = 0.1, zorder = -1 )
    plt.setp ( a.spines.values(), linewidth = 1.5 )

# 🔹 CSS para tooltips flotantes
tooltip_css = """
    <style>
    .tooltip-container {
        position: relative;
        display: flex;
        align-items: center;  /* Alinear verticalmente */
        gap: 5px; /* Espacio entre icono y texto */
        font-size: 14px;  /* Tamaño del texto */
        margin-bottom: -100px; /* Reduce espacio entre el label y el slider */
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        color: blue;
        font-weight: bold;
        margin-left: 5px;
        font-size: 16px;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        #background-color: black;
        #color: #fff;
        background-color: #31333F; /* Mismo color que el sidebar de Streamlit */
        color: white;
        text-align: center;
        border-radius: 5px;
        padding: 10px;
        border: 2px solid #666; /* Borde para resaltar */
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 1000;
        opacity: 0;
        transition: opacity 0.3s;
        margin: 10px 20px 10px 20px; /* Margen en todas las direcciones */
        font-size: 14px;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
"""

# Configurar la página
st.set_page_config(page_title="Galaxy Cluster Analysis", layout="wide")

# Insertar CSS en Streamlit
st.markdown(tooltip_css, unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.title("Control Panel")

# Selección de variables
i_options = [1, 2, 3, 4, 5]
model_type_options = ['balanced_random_forest', 'support_vector_machine', 'xgboost', 'lightgbm', 'mlpclassifier', 'assembled']
dim_reduction_options = ['pca', 'umap', 'umap_supervised', 'None']

i = st.sidebar.selectbox("Select Features", i_options, index=0)
dim_reduction = st.sidebar.selectbox("Select Dimensionality Reduction", dim_reduction_options)
model_type = st.sidebar.selectbox("Select Model Type", model_type_options)

# # Condición para fijar dim_reduction en None si "all" está seleccionado
# if model_type == "assembled":
#     dim_reduction = "None"  # Fijar en None
# else:
#     dim_reduction = st.sidebar.selectbox("Select Dimensionality Reduction", dim_reduction_options)

st.sidebar.markdown("---")

# Barras deslizantes para ajustar parámetros
# 🔹 Tooltip para "Threshold for flag_member2"
st.sidebar.markdown("""
    <div class="tooltip-container">
        <span>Threshold for flag_member2</span>
        <div class="tooltip"> ℹ️
            <span class="tooltiptext">
                The definition of <b>flag_member2</b> is related to <b>prob1</b>. For a given value of <b>prob1 = X</b>, <b>flag_member2</b> is calculated as follows:<br><br>
                <div style="text-align: center;">
                    <table class="tooltip-table" style="margin: auto;">
                        <tr>
                            <th>Condition</th>
                            <th>flag_member2</th>
                        </tr>
                        <tr>
                            <td><b>prob1 > X</b></td>
                            <td style="color:lightcoral;"><b>1 (Member)</b></td>
                        </tr>
                        <tr>
                            <td><b>prob1 < X</b></td>
                            <td style="color:lightblue;"><b>0 (Outlier)</b></td>
                        </tr>
                    </table>
                </div>
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)
flag_member2_th = st.sidebar.slider("", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
odds_th = st.sidebar.slider("Odds", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
R_FACTOR = st.sidebar.slider("R200 factor", min_value=0.1, max_value=10.0, value=14.0, step=0.1)

st.sidebar.markdown("""
    <div class="tooltip-container">
        <span>prob1</span>
        <div class="tooltip"> ℹ️
            <span class="tooltiptext">Depending on the selected model, <b>prob1</b> is the probability of each object being a member of the cluster.</span>
        </div>
    </div>
""", unsafe_allow_html=True)
prob1_th = st.sidebar.slider("", min_value=0.05, max_value=1.0, value=0.5, step=0.05)

st.sidebar.markdown("---")

show_scatter_members = st.sidebar.checkbox("members", value=True)
show_scatter_background = st.sidebar.checkbox("background", value=True)

@st.cache_data
def load_static_data(filename):
    """Carga datos estáticos solo una vez (cacheados)."""
    return pd.read_csv(filename)

# Cargar el archivo estático (cacheado)
df_output_parameters = load_static_data(data_dir + 'results/df_output_parameters.csv') 

@st.cache_data
def load_dynamic_data(i, dim_reduction, model_type):
    """Carga datos dinámicos basados en la selección del usuario."""
    filename = data_dir + f'results/ZML_{i}-{dim_reduction}-{model_type}.csv'
    return pd.read_csv(filename)

# Cargar el archivo dinámico
r = load_dynamic_data(i, dim_reduction, model_type)

# ---- MAIN CONTENT ----
st.title("Galaxy Cluster Membership Analysis")
st.write(f"### Features: {i} | Dimensionality Reduction: {dim_reduction} | Model: {model_type}")

# Crear diseño de tres columnas para la primera fila
col1, col2, col3 = st.columns(3)

# 📌 Gráfico en Columna 1
with col1:
    st.write("🖼️ Metrics evaluation (PNG Image)")
    figure_path = data_dir + f"figures/ROC_{i}-{dim_reduction}-{model_type}.png"
    if dim_reduction == 'None':
        st.warning("⚠️ No dimensionality reduction applied. No figure available.")
    elif not os.path.exists(figure_path):
        st.warning(f"⚠️ The figure for {dim_reduction} - {model_type} is not found.")
    else:
        st.image(figure_path, caption=f"Figure for {dim_reduction} - {model_type}", use_container_width=True)

# 📌 Gráfico en Columna 2
with col2:
    st.write("🖼️ Dimnesionality reduction (PNG Image)")
    figure_path = data_dir + f"figures/DIM_{i}-{dim_reduction}-{model_type}.png"
    if dim_reduction == 'None':
        st.warning("⚠️ No dimensionality reduction applied. No figure available.")
    elif not os.path.exists(figure_path):
        st.warning(f"⚠️ The figure for {dim_reduction} - {model_type} is not found.")
    else:
        st.image(figure_path, caption=f"Figure for {dim_reduction} - {model_type}", use_container_width=True)

# 📌 Gráfico interactivo en Columna 3 (Custom Graph)
with col3:
    st.write("📊 Comparision")
    # Definir las condiciones y asignar flag_member2
    conditions = [(r.prob1 > flag_member2_th), (r.prob1 < flag_member2_th)]
    choices = [0, 1]
    r['flag_member2'] = np.select(conditions, choices, default=-1)

    # Filtrar por cluster
    rr = r.query("name == 'MKW4'")

    # ---- PLOTEO ----
    size = 80
    fig3, (ax0, ax1) = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))

    x = 'radius_Mpc'
    y = 'zml' 

    # 📌 PLOT 1: Photometric Membership
    df_plot = rr.query("((flag_member == 0) & (odds > @odds_th))").dropna(subset=[x, y])
    ax0.scatter(x, y, data=df_plot, s=size * 2.2, alpha=0.5, edgecolors='white', color='red', marker='.', zorder=2, label=f'flag_member == 0 ({len(df_plot)})')

    df_plot = rr.query("((flag_member == 1) & (odds > @odds_th))").dropna(subset=[x, y])
    ax0.scatter(x, y, data=df_plot, s=size * 2.2, alpha=0.7, edgecolors='white', color='blue', marker='.', zorder=1, label=f'flag_member == 1 ({len(df_plot)})')

    ax0.set_title(f"Photometric redshift x distance (odds > {odds_th})", fontsize=10)
    ax0.set_xlabel('radius_Mpc')
    ax0.set_ylabel(y)
    custom_axes(ax0)
    ax0.legend()

    # 📌 PLOT 2: Flag Membership
    df_plot = rr[rr["D_CENTER/R200_deg"] < R_FACTOR].query("(flag_member2 == 0)").dropna(subset=[x, y])
    ax1.scatter(x, y, data=df_plot, s=size * 2.2, alpha=0.5, edgecolors='white', color='red', marker='.', zorder=2, label=f'flag_member2 == 0 ({len(df_plot)})')

    df_plot = rr[rr["D_CENTER/R200_deg"] < R_FACTOR].query("(flag_member2 == 1)").dropna(subset=[x, y])
    ax1.scatter(x, y, data=df_plot, s=size * 2.2, alpha=0.7, edgecolors='white', color='blue', marker='.', zorder=1, label=f'flag_member2 == 1 ({len(df_plot)})')
    xlim = ax1.get_xlim(); ylim = ax1.get_ylim()
    ax0.set_xlim(xlim); ax0.set_ylim(ylim); 

    ax1.axvline(x=5.204, color='black', linestyle='--', zorder=1)
    ax1.axhline(y=0.018, color='black', linestyle='--', zorder=1)
    ax1.axhline(y=0.036, color='black', linestyle='--', zorder=1)

    ax1.set_xlabel('radius_Mpc')
    ax1.set_ylabel(y)
    custom_axes(ax1)
    ax1.legend(loc='best', markerscale=1)

    fig3.suptitle(f"Evaluación del Modelo: i = {i}, dim_reduction = {dim_reduction}, model_type = {model_type}", fontsize=10, y=0.97)
    fig3.tight_layout()

    # Mostrar en Streamlit
    st.pyplot(fig3)

# Crear segunda fila de gráficos
col4, col5, col6 = st.columns(3)

# 📌 Gráfico en Columna 4
with col4:
    st.write("📊 CMD")

    fig4 = plt.figure(figsize=(20, 6))  
    gs = GridSpec(1, 2, width_ratios=[1, 1]) 
    ax0 = fig4.add_subplot(gs[0, 0])

    x = 'r_auto'; y = 'g_auto-r_auto'
    df_plot = rr[rr["D_CENTER/R200_deg"] < R_FACTOR].query("label == 'candidates' & prob1 > @prob1_th")
    x_min = rr.query("label == 'candidates'").r_auto.min(); x_max = rr.query("label == 'candidates'").r_auto.max()
    scatter2 = ax0.scatter ( x, y, data = df_plot, s = size*1.2, c=df_plot.prob1, edgecolors = 'white', alpha = 0.9, 
                        cmap=plt.get_cmap('jet', 11), label = 'candidates (%i)' % (len(df_plot)), vmin=0, vmax=1, zorder = 1 )   
    xlim = ax0.get_xlim(); ylim = ax0.get_ylim()

    if show_scatter_background:
        df_plot = rr.query("label == 'background'")
        ax0.scatter ( x, y, data = df_plot, s = size*3.2, alpha = 0.9, edgecolors = 'black', color = 'purple', marker = '.', 
                    linewidth = 1.2, label = 'background (%i)' % (len(df_plot)), zorder = 2 )  

    if show_scatter_members:
        df_plot = rr.query("label == 'members'")
        ax0.scatter ( x, y, data = df_plot, s = size*3.2, alpha = 0.9, edgecolors = 'black', color = 'red', marker = '.', 
                    linewidth = 1.2, label = 'members (%i)' % (len(df_plot)), zorder = 3 )  

    # Ajustar las posiciones de los gráficos para dejar espacio para la barra de color
    fig4.tight_layout(pad=2.0)
    pos_ax2 = ax0.get_position()  # Posición del segundo gráfico

    # Añadir barra de color (colorbar) con mismo alto que los gráficos
    cbar_ax = fig4.add_axes([
        pos_ax2.x1 + 0.03,  # Alinear a la derecha del segundo gráfico
        pos_ax2.y0,         # Parte inferior alineada con el gráfico
        0.02,               # Ancho de la barra de color
        pos_ax2.height      # Altura igual a la del gráfico
    ])
    cbar = plt.colorbar(scatter2, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Probability (prob1)', labelpad=10)
    cbar.minorticks_on() 

    for a in [ ax0 ]:
        handles, labels = a.get_legend_handles_labels()
        custom_axes(a); a.set_ylim (-1, 2); a.set_xlim(x_min, x_max)
        a.invert_xaxis(); a.set_xlabel ( "r [mag]" ); a.set_ylabel ( "(g-r) [mag]" )
        for item in ([a.title, a.xaxis.label, a.yaxis.label] + a.get_xticklabels() + a.get_yticklabels()): item.set_fontsize(20)
        lgd = a.legend ( handles, labels, fontsize = 10, framealpha = 1, loc = 'lower right', ncol = 1, borderaxespad = 1. )  
    plt.tight_layout()
    st.pyplot(fig4)

# 📌 Gráfico en Columna 5
with col5:
    st.write("📊 REFF-MAG")

    fig5 = plt.figure(figsize=(20, 6))  
    gs = GridSpec(1, 2, width_ratios=[1, 1]) 

    ax0 = fig5.add_subplot(gs[0, 0])

    x = 'r_auto'; y = 'FLUX_RADIUS_50'
    df_plot = rr[rr["D_CENTER/R200_deg"] < R_FACTOR].query("label == 'candidates' & prob1 > @prob1_th")
    x_min = rr.query("label == 'candidates'").r_auto.min(); x_max = rr.query("label == 'candidates'").r_auto.max()
    y_min = rr.query("label == 'candidates'").FLUX_RADIUS_50.min(); y_max = rr.query("label == 'candidates'").FLUX_RADIUS_50.max()
    scatter2 = ax0.scatter ( x, y, data = df_plot, s = size*1.2, c=df_plot.prob1, edgecolors = 'white', alpha = 0.9, 
                            cmap=plt.get_cmap('jet', 11), label = 'candidates (%i)' % (len(df_plot)), vmin=0, vmax=1, zorder = 1 )   
    xlim = ax0.get_xlim(); ylim = ax0.get_ylim()

    if show_scatter_background:
        df_plot = rr.query("label == 'background'")
        ax0.scatter ( x, y, data = df_plot, s = size*3.2, alpha = 0.9, edgecolors = 'black', color = 'purple', marker = '.', 
                    linewidth = 1.2, label = 'background (%i)' % (len(df_plot)), zorder = 2 )  

    if show_scatter_members:
        df_plot = rr.query("label == 'members'")
        ax0.scatter ( x, y, data = df_plot, s = size*3.2, alpha = 0.9, edgecolors = 'black', color = 'red', marker = '.', 
                    linewidth = 1.2, label = 'members (%i)' % (len(df_plot)), zorder = 4 )  

    # Ajustar las posiciones de los gráficos para dejar espacio para la barra de color
    fig5.tight_layout(pad=2.0)
    pos_ax2 = ax0.get_position()  # Posición del segundo gráfico

    # Añadir barra de color (colorbar) con mismo alto que los gráficos
    cbar_ax = fig5.add_axes([
        pos_ax2.x1 + 0.03,  # Alinear a la derecha del segundo gráfico
        pos_ax2.y0,         # Parte inferior alineada con el gráfico
        0.02,               # Ancho de la barra de color
        pos_ax2.height      # Altura igual a la del gráfico
    ])
    cbar = plt.colorbar(scatter2, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Probability (prob1)', labelpad=10)
    cbar.minorticks_on() 

    for a in [ ax0 ]:
        handles, labels = a.get_legend_handles_labels()
        custom_axes ( a ); a.invert_xaxis(); a.set_yscale('log')
        a.set_yscale('log'); a.set_ylim(y_min, y_max); a.set_xlim(x_min, x_max); a.invert_xaxis()
        a.set_xlabel("r [mag]"); a.set_ylabel("R$_{eff}$ [kpc]")
        for item in ([a.title, a.xaxis.label, a.yaxis.label] + a.get_xticklabels() + a.get_yticklabels()): item.set_fontsize(20)
        a.legend ( handles, labels, fontsize = 10, framealpha = 1, loc = 'lower right', ncol = 1, borderaxespad = 1. )  

    plt.tight_layout()
    plt.show()
    st.pyplot(fig5)

# 📌 Gráfico en Columna 6
with col6:
    st.write("📊 RA-DEC")

    fig6 = plt.figure ( figsize = (15, 10) )
    center = SkyCoord ( ra=180.9884, dec=+1.8883,  frame = FK5, unit = u.deg )

    R200 = 3.566/5
    ax0 = plt.axes ( projection = 'astro zoom', center = center, radius = 9*u.deg )

    x = 'ra'; y = 'dec'
    df_plot = rr[rr["D_CENTER/R200_deg"] < R_FACTOR].query("label == 'candidates' & prob1 > @prob1_th").dropna(subset=['prob1'])
    scatter2 = ax0.scatter ( x, y, data = df_plot, s = size*1.5, c=df_plot.prob1, edgecolors = 'white', alpha = 0.9,  cmap=plt.get_cmap('jet', 11),
                linewidth = 0.4, transform = ax0.get_transform('world'), label = 'candidates (%s)' % (len(df_plot)), zorder = 1 )   
    xlim = ax0.get_xlim(); ylim = ax0.get_ylim()

    for R_FACTOR in [ 1, 5 ]:
        R_200 = SphericalCircle ( (center.ra.deg * u.degree , center.dec.deg * u.degree ), R_FACTOR * R200 * u.degree, 
                                edgecolor = 'black', facecolor = 'none', linewidth = 3.0, linestyle = '--', 
                                transform = ax0.get_transform('world'), zorder = 2 )
        ax0.add_patch(R_200) 

    if show_scatter_background:
        df_plot = rr.query("label == 'background'")
        ax0.scatter ( x, y, data = df_plot, s = size*3.5, alpha = 0.9, edgecolors = 'black', color = 'purple', marker = '.', 
                    linewidth = 1.2,transform = ax0.get_transform('world'), label = 'background (%i)' % (len(df_plot)), zorder = 2 )  

    if show_scatter_members:
        df_plot = rr.query("label == 'members'")
        ax0.scatter ( x, y, data = df_plot, s = size*3.5, alpha = 0.9, edgecolors = 'black', color = 'red', marker = '.', 
                    linewidth = 1.2, transform = ax0.get_transform('world'), label = 'members (%i)' % (len(df_plot)), zorder = 4 )  

    # Ajustar las posiciones de los gráficos para dejar espacio para la barra de color
    fig6.tight_layout(pad=2.0)
    pos_ax2 = ax0.get_position()  # Posición del segundo gráfico

    # Añadir barra de color (colorbar) con mismo alto que los gráficos
    cbar_ax = fig6.add_axes([
        pos_ax2.x1 + 0.17,  # Alinear a la derecha del segundo gráfico
        pos_ax2.y0,         # Parte inferior alineada con el gráfico
        0.02,               # Ancho de la barra de color
        pos_ax2.height      # Altura igual a la del gráfico
    ])

    cbar = plt.colorbar(scatter2, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Probability (prob1)', labelpad=10)
    cbar.minorticks_on() 

    ax0.set_xlim ( 60, 730 ); ax0.set_ylim ( 150, 550 )
    ax0.grid()
    ax0.set_xlabel ( "Right Ascension [J2000]" ); ax0.set_ylabel ( "Declination [J2000]" )
    leg = plt.legend ( loc = 'lower right' )
    plt.show()
    st.pyplot(fig6)

st.markdown("---")

st.title("📊 Parameters")
filtered_df = df_output_parameters[
    (df_output_parameters["i"] == i) & 
    (df_output_parameters["dim_reduction"].isin(dim_reduction_options)) & 
    (df_output_parameters["model_type"].isin(model_type_options))
    ]

st.dataframe(filtered_df.reset_index(drop=True).head(10), height=600, use_container_width=True)

st.markdown("---")

st.title("📊 Notes")


geom = [ 'A', 'B', 'THETA', 'ELLIPTICITY', 'ELONGATION', 'PETRO_RADIUS', 'FLUX_RADIUS_50', 'FLUX_RADIUS_90', 'MU_MAX_g', 'MU_MAX_r', 'BACKGROUND_g', 'BACKGROUND_r', 's2n_g_auto', 's2n_r_auto' ]
ngeom = [ 'D_CENTER/R200_deg', '(A/B)','(FLUX_RADIUS_50/PETRO_RADIUS)', '(FLUX_RADIUS_90/PETRO_RADIUS)', 
           '(FLUX_RADIUS_50/PETRO_RADIUS)*(A/B)', 'Densidad_vecinos', 'r_auto/area', 'Area_Voronoi', 'Area_Voronoi_norm', 'Diferencia_angular' ]

bands = [ 'J0378_auto', 'J0395_auto', 'J0410_auto', 'J0430_auto', 'J0515_auto', 'J0660_auto', 'J0861_auto', 'g_auto', 'i_auto', 'r_auto', 'u_auto', 'z_auto' ]
bands_e = [ 'e_J0378_auto', 'e_J0395_auto', 'e_J0410_auto', 'e_J0430_auto', 'e_J0515_auto', 'e_J0660_auto', 'e_J0861_auto', 'e_g_auto', 'e_i_auto', 'e_r_auto', 'e_u_auto', 'e_z_auto' ]
bands_PS = [ 'J0378_PStotal', 'J0395_PStotal', 'J0410_PStotal', 'J0430_PStotal', 'J0515_PStotal', 'J0660_PStotal', 'J0861_PStotal', 'g_PStotal', 'i_PStotal', 'r_PStotal', 'u_PStotal', 'z_PStotal' ] 
bands_PS_e = [ 'e_J0378_PStotal', 'e_J0395_PStotal', 'e_J0410_PStotal', 'e_J0430_PStotal', 'e_J0515_PStotal', 'e_J0660_PStotal', 'e_J0861_PStotal', 'e_g_PStotal', 'e_i_PStotal', 'e_r_PStotal', 'e_u_PStotal', 'e_z_PStotal' ]
band_iso = [ 'r_iso', 'r_petro', 'r_aper_3', 'r_aper_6' ]

features = {}

features[1] = bands 
features[2] = ngeom 
features[3] = bands + geom 
features[4] = bands + ngeom 
features[5] = bands + ngeom + geom + bands_e + bands_PS + bands_PS_e + band_iso
