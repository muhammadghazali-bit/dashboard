import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Halaman Dashboard
st.set_page_config(page_title="Dashboard Fiskal 97 Kota", page_icon="📊", layout="wide")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    df = pd.read_csv("data_fiskal.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'data_fiskal.csv' tidak ditemukan. Pastikan file tersebut berada di folder yang sama dengan app.py.")
    st.stop()

# ================================
# SIDEBAR: FILTER DATA
# ================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/320/320113.png", width=100) # Ikon opsional
st.sidebar.header("🛠️ Panel Filter")

# Filter Tahun
years = df['Tahun'].unique().tolist()
min_year, max_year = min(years), max(years)
selected_years = st.sidebar.slider("Pilih Rentang Tahun", min_value=min_year, max_value=max_year, value=(min_year, max_year))

# Filter Kota
cities = df['Kota'].unique().tolist()
# Secara default, pilih 5 kota pertama agar grafik tidak terlalu padat di awal
selected_cities = st.sidebar.multiselect("Pilih Kota", options=cities, default=cities[:5]) 

# Data yang telah difilter
filtered_df = df[(df['Tahun'] >= selected_years[0]) & 
                 (df['Tahun'] <= selected_years[1]) & 
                 (df['Kota'].isin(selected_cities))]

# Pengecekan data kosong
if filtered_df.empty:
    st.warning("Data tidak tersedia untuk kombinasi filter yang dipilih.")
    st.stop()

# ================================
# MAIN LAYOUT: TABS
# ================================
st.title("📊 Dashboard Data Fiskal & Makroekonomi 97 Kota")
st.markdown("Dashboard ini menyajikan metrik visual dan analisis statistik (Regresi & Uji Asumsi) berdasarkan `data_fiskal.csv`.")

tab1, tab2, tab3 = st.tabs(["📈 Visualisasi & Tren", "🔬 Analisis Regresi & Uji Asumsi", "🗄️ Tabel Data Mentah"])

# ---------------------------------
# TAB 1: VISUALISASI & TREN
# ---------------------------------
with tab1:
    st.subheader("Indikator Makro Terpilih (Rata-rata)")
    
    # Menampilkan KPI (Key Performance Indicators)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label="Rata-rata Pertumbuhan Ekonomi", value=f"{filtered_df['PE'].mean():.2f}%")
    kpi2.metric(label="Rata-rata IPM", value=f"{filtered_df['IPM'].mean():.2f}")
    kpi3.metric(label="Rata-rata Kemiskinan", value=f"{filtered_df['KEMISKINAN'].mean():.2f}%")
    kpi4.metric(label="Total Observasi Terpilih", value=f"{len(filtered_df)} Data")
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**1. Tren Variabel dari Waktu ke Waktu**")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if 'Tahun' in numeric_cols: numeric_cols.remove('Tahun')
        
        trend_var = st.selectbox("Pilih Variabel untuk Line Chart", numeric_cols, index=numeric_cols.index("PAD") if "PAD" in numeric_cols else 0)
        fig_trend = px.line(filtered_df, x='Tahun', y=trend_var, color='Kota', markers=True, title=f"Tren {trend_var} (Berdasarkan Kota)")
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_chart2:
        st.markdown("**2. Perbandingan Variabel Antar Kota (Bar Chart)**")
        bar_var = st.selectbox("Pilih Variabel untuk Bar Chart", numeric_cols, index=numeric_cols.index("IPM") if "IPM" in numeric_cols else 0)
        
        # Agregasi mean berdasarkan kota
        bar_data = filtered_df.groupby('Kota')[bar_var].mean().reset_index().sort_values(by=bar_var, ascending=False)
        fig_bar = px.bar(bar_data, x='Kota', y=bar_var, color='Kota', title=f"Rata-rata {bar_var} per Kota")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    st.markdown("**3. Scatter Plot: Hubungan Antar Dua Variabel**")
    scat_col1, scat_col2 = st.columns(2)
    with scat_col1:
        x_var = st.selectbox("Pilih Variabel X (Sumbu Horizontal)", numeric_cols, index=numeric_cols.index("PAD") if "PAD" in numeric_cols else 0)
    with scat_col2:
        y_var = st.selectbox("Pilih Variabel Y (Sumbu Vertikal)", numeric_cols, index=numeric_cols.index("PE") if "PE" in numeric_cols else 1)
    
    fig_scatter = px.scatter(filtered_df, x=x_var, y=y_var, color='Kota', size_max=10, hover_data=['Tahun'], 
                             title=f"Korelasi: {x_var} vs {y_var}")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------
# TAB 2: ANALISIS REGRESI & ASUMSI
# ---------------------------------
with tab2:
    st.subheader("Analisis Regresi OLS & Uji Formalitas (Asumsi Klasik)")
    
    st.info("💡 **Tips:** Untuk regresi yang lebih akurat dan terhindar dari bias, Anda dapat memilih untuk menggunakan seluruh populasi dataset (tanpa filter sidebar) atau menggunakan data yang sudah Anda filter.")
    
    use_filtered = st.checkbox("Gunakan Data yang Difilter (Pastikan jumlah observasi (N) mencukupi)", value=False)
    reg_df = filtered_df if use_filtered else df
    
    reg_c1, reg_c2 = st.columns(2)
    with reg_c1:
        y_reg = st.selectbox("Pilih Variabel Dependen (Y) - Variabel Terikat", numeric_cols, index=numeric_cols.index("PE") if "PE" in numeric_cols else 0)
    with reg_c2:
        x_reg = st.multiselect("Pilih Variabel Independen (X) - Bebas", numeric_cols, default=["PAD", "DAU_Grant", "DAK_Fisik"])
        
    if st.button("Jalankan Uji Regresi", type="primary"):
        if len(x_reg) == 0:
            st.error("Silakan pilih minimal satu variabel independen (X) untuk diuji.")
        else:
            # Pembangunan Model
            X = reg_df[x_reg]
            X = sm.add_constant(X) # Menambahkan intercept
            Y = reg_df[y_reg]
            
            # Membersihkan missing values jika ada
            model_data = pd.concat([Y, X], axis=1).dropna()
            Y_clean = model_data[y_reg]
            X_clean = model_data.drop(columns=[y_reg])
            
            model = sm.OLS(Y_clean, X_clean).fit()
            
            st.markdown("### 1. Ringkasan Regresi (Statistik)")
            # Menggunakan st.text agar format summary rata (monospace)
            st.text(model.summary())
            
            # --------------------------------
            # INTERPRETASI REGRESI
            # --------------------------------
            st.markdown("#### 🧠 Interpretasi Model Regresi:")
            r_sq = model.rsquared
            p_f = model.f_pvalue
            
            # R-Squared
            st.write(f"- **Koefisien Determinasi ($R^2$):** {r_sq:.4f}. Artinya, sebesar **{r_sq*100:.2f}%** variabilitas pada **{y_reg}** berhasil dijelaskan oleh model (variabel independen). Sisanya {100-(r_sq*100):.2f}% dijelaskan oleh faktor di luar model.")
            
            # Uji F
            if p_f < 0.05:
                st.write(f"- **Uji F (Signifikansi Simultan):** P-Value ({p_f:.4e}) < 0.05. Secara **bersama-sama**, seluruh variabel independen **berpengaruh signifikan** terhadap **{y_reg}**.")
            else:
                st.write(f"- **Uji F (Signifikansi Simultan):** P-Value ({p_f:.4e}) > 0.05. Secara bersama-sama, variabel independen **tidak berpengaruh signifikan**.")
                
            # Uji T (Parsial)
            st.write("- **Uji T (Signifikansi Parsial):**")
            for col in x_reg:
                p_t = model.pvalues[col]
                coef = model.params[col]
                if p_t < 0.05:
                    st.write(f"  - **{col}**: Berpengaruh **Signifikan** (P-value={p_t:.4f}). Jika {col} naik 1 satuan, maka {y_reg} akan berubah sebesar **{coef:.4f}**, asumsi variabel lain konstan (Ceteris Paribus).")
                else:
                    st.write(f"  - **{col}**: **Tidak** berpengaruh signifikan (P-value={p_t:.4f}).")

            st.markdown("---")
            st.markdown("### 2. Uji Formalitas (Asumsi Klasik)")
            residuals = model.resid
            
            c_asum1, c_asum2 = st.columns(2)
            
            with c_asum1:
                # 1. Normalitas
                st.markdown("**A. Uji Normalitas (Shapiro-Wilk)**")
                stat_sw, p_sw = stats.shapiro(residuals)
                st.write(f"- **P-value:** {p_sw:.4f}")
                if p_sw > 0.05:
                    st.success("✔ Asumsi Terpenuhi: Residual terdistribusi normal.")
                else:
                    st.error("✖ Asumsi Dilanggar: Residual tidak terdistribusi normal. (Solusi: Lakukan transformasi log/ln atau hapus outlier).")
                
                # Visualisasi Residual
                fig, ax = plt.subplots(figsize=(4,2))
                sns.histplot(residuals, kde=True, ax=ax, color='blue')
                ax.set_title("Histogram Residual")
                st.pyplot(fig)
                
                # 3. Autokorelasi
                st.markdown("**C. Uji Autokorelasi (Durbin-Watson)**")
                dw = sm.stats.stattools.durbin_watson(residuals)
                st.write(f"- **Nilai D-W:** {dw:.4f}")
                if 1.5 < dw < 2.5:
                    st.success("✔ Asumsi Terpenuhi: Tidak ditemukan gejala autokorelasi.")
                else:
                    st.error("✖ Asumsi Dilanggar: Ditemukan autokorelasi. (Biasa terjadi pada data panel/runtut waktu).")

            with c_asum2:
                # 2. Heteroskedastisitas
                st.markdown("**B. Uji Heteroskedastisitas (Breusch-Pagan)**")
                bp_test = het_breuschpagan(residuals, X_clean)
                st.write(f"- **P-value:** {bp_test[1]:.4f}")
                if bp_test[1] > 0.05:
                    st.success("✔ Asumsi Terpenuhi: Varians residual konstan (Homoskedastisitas).")
                else:
                    st.error("✖ Asumsi Dilanggar: Terjadi Heteroskedastisitas. (Solusi: Gunakan *Robust Standard Error*).")
                    
                # 4. Multikolinearitas
                st.markdown("**D. Uji Multikolinearitas (VIF)**")
                if len(x_reg) > 1:
                    vif_data = pd.DataFrame()
                    vif_data["Variabel"] = X_clean.columns
                    vif_data["VIF"] = [variance_inflation_factor(X_clean.values, i) for i in range(X_clean.shape[1])]
                    vif_data = vif_data[vif_data["Variabel"] != "const"]
                    st.dataframe(vif_data, use_container_width=True)
                    
                    if (vif_data["VIF"] > 10).any():
                        st.error("✖ Asumsi Dilanggar: Terdapat variabel dengan VIF > 10, mengindikasikan adanya korelasi kuat antar variabel independen.")
                    else:
                        st.success("✔ Asumsi Terpenuhi: Tidak ada gejala Multikolinearitas (Semua nilai VIF < 10).")
                else:
                    st.info("Uji Multikolinearitas (VIF) memerlukan minimal 2 variabel independen.")

# ---------------------------------
# TAB 3: DATA MENTAH
# ---------------------------------
with tab3:
    st.subheader(f"Tabel Data Mentah (Hasil Filter)")
    st.write(f"Menampilkan **{len(filtered_df)}** baris data berdasarkan filter yang dipilih.")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Tombol Unduh Data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Data sebagai CSV",
        data=csv,
        file_name='data_fiskal_filtered.csv',
        mime='text/csv',
    )