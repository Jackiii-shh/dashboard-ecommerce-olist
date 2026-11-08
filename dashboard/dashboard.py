import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ----------------------------------------------------------------------------
# Konfigurasi halaman & gaya visual (konsisten dengan notebook)
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard E-Commerce Olist", page_icon=":bar_chart:", layout="wide")
sns.set_theme(style="whitegrid")

ACCENT = "#2E6F9E"  # biru   -> sorotan utama
MUTED = "#C9CED6"   # abu    -> pembanding
WARN = "#E07A1F"    # oranye -> penurunan / perhatian

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_data.csv")


def fmt_brl(x, pos=None):
    if abs(x) >= 1e6:
        return f"R$ {x / 1e6:.2f}M"
    if abs(x) >= 1e4:
        return f"R$ {x / 1e3:.0f}K"
    if abs(x) >= 1e3:
        return f"R$ {x / 1e3:.1f}K"
    return f"R$ {x:.0f}"


def fmt_axis(x, pos=None):
    if abs(x) >= 1e6:
        return f"R$ {x / 1e6:g}M"
    if abs(x) >= 1e3:
        return f"R$ {x / 1e3:g}K"
    return f"R$ {x:g}"


def fmt_share(s):
    return f"{s:.3f}%" if s < 0.1 else f"{s:.1f}%"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["order_purchase_timestamp"])
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M")
    return df


def safe_qcut(series, labels):
    """qcut dengan fallback jika batas bin tidak unik (data hasil filter kecil)."""
    try:
        return pd.qcut(series, 5, labels=labels)
    except ValueError:
        return pd.qcut(series.rank(method="first"), 5, labels=labels)


def assign_segment(row):
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if f >= 2 and r >= 4:
        return "Champions"
    if f >= 2:
        return "Repeat Customers"
    if r >= 4 and m >= 4:
        return "New High-Value"
    if r >= 4:
        return "Recent Buyers"
    if r <= 2 and m >= 4:
        return "At-Risk High-Value"
    if r <= 2:
        return "Lost / Dormant"
    return "Need Attention"


def build_rfm(df):
    reference_date = df["order_purchase_timestamp"].max().normalize() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_unique_id").agg(
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("price", "sum"),
    ).reset_index()
    rfm["recency"] = (reference_date - rfm["last_purchase"]).dt.days
    rfm["R_score"] = safe_qcut(rfm["recency"], [5, 4, 3, 2, 1]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["F_score"] = pd.cut(rfm["frequency"], bins=[0, 1, 2, np.inf], labels=[1, 2, 3]).astype(int)
    rfm["segment"] = rfm.apply(assign_segment, axis=1)
    return rfm, reference_date


# ----------------------------------------------------------------------------
# Fungsi visualisasi
# ----------------------------------------------------------------------------
def plot_categories(summary, n):
    top = summary.head(n).sort_values("revenue")
    bottom = summary.tail(n).sort_values("revenue")
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4.5, 0.5 * n + 2)))
    hl = min(3, n)
    specs = [
        (axes[0], top, [MUTED] * (len(top) - hl) + [ACCENT] * hl, f"{n} Kategori dengan Revenue Tertinggi", "Total revenue (R$)"),
        (axes[1], bottom, [WARN] * hl + [MUTED] * (len(bottom) - hl), f"{n} Kategori dengan Revenue Terendah",
         "Total revenue (R$) - skala berbeda dari panel kiri"),
    ]
    for ax, data, colors, title, xlabel in specs:
        bars = ax.barh(data.index, data["revenue"], color=colors)
        ax.bar_label(
            bars,
            labels=[f"{fmt_brl(r)}  ({fmt_share(s)})" for r, s in zip(data["revenue"], data["revenue_share_pct"])],
            padding=4, fontsize=9,
        )
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(fmt_axis))
        ax.set_xlim(0, max(data["revenue"].max(), 1) * 1.38)
        ax.grid(axis="y", visible=False)
        sns.despine(ax=ax, left=True)
    plt.tight_layout()
    return fig


def plot_trend(monthly):
    x = monthly.index.to_timestamp()
    peak = monthly["revenue"].idxmax()
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1.4]})

    ax = axes[0]
    ax.plot(x, monthly["revenue"], marker="o", color=ACCENT, linewidth=2.5)
    ax.scatter([peak.to_timestamp()], [monthly.loc[peak, "revenue"]], s=140, color=WARN, zorder=5)
    ax.annotate(f"Puncak: {peak.strftime('%b %Y')}\n{fmt_brl(monthly.loc[peak, 'revenue'])}",
                xy=(peak.to_timestamp(), monthly.loc[peak, "revenue"]), xytext=(-18, 0), textcoords="offset points",
                ha="right", va="center", fontsize=10, fontweight="bold", color=WARN)
    ax.set_title("Revenue bulanan", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Revenue (R$)")
    ax.set_ylim(0, monthly["revenue"].max() * 1.15)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_axis))

    ax = axes[1]
    ax.plot(x, monthly["total_orders"], marker="o", color=ACCENT, linewidth=2.5)
    ax.scatter([peak.to_timestamp()], [monthly.loc[peak, "total_orders"]], s=140, color=WARN, zorder=5)
    ax.set_title("Jumlah pesanan bulanan", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Jumlah pesanan")
    ax.set_ylim(0, monthly["total_orders"].max() * 1.15)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    ax = axes[2]
    mom = monthly["revenue_mom_pct"].iloc[1:]
    if len(mom):
        ax.bar(mom.index.to_timestamp(), mom, width=20, color=[ACCENT if v >= 0 else WARN for v in mom])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Pertumbuhan revenue Month-over-Month (biru = naik, oranye = turun)", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("MoM (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlabel("Bulan transaksi")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.tick_params(axis="x", labelsize=9)
    for a in axes:
        sns.despine(ax=a)
    plt.tight_layout()
    return fig


def plot_rfm(seg):
    order = seg.sort_values("customers").index
    risk = ("At-Risk High-Value", "Lost / Dormant")
    colors = [WARN if s in risk else ACCENT for s in order]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    bars = axes[0].barh(order, seg.loc[order, "customers"], color=colors)
    axes[0].bar_label(bars, labels=[f"{v:,.0f} ({p:.1f}%)" for v, p in zip(seg.loc[order, "customers"], seg.loc[order, "customer_share_pct"])],
                      padding=4, fontsize=9)
    axes[0].set_title("Jumlah pelanggan per segmen", loc="left", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Jumlah pelanggan unik")
    axes[0].set_xlim(0, seg["customers"].max() * 1.28)
    axes[0].xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    bars = axes[1].barh(order, seg.loc[order, "avg_monetary"], color=colors)
    axes[1].bar_label(bars, labels=[f"R$ {v:,.0f}" for v in seg.loc[order, "avg_monetary"]], padding=4, fontsize=9)
    axes[1].set_title("Rata-rata total belanja per pelanggan", loc="left", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Rata-rata monetary (R$)")
    axes[1].set_xlim(0, seg["avg_monetary"].max() * 1.28)
    for a in axes:
        a.grid(axis="y", visible=False)
        sns.despine(ax=a, left=True)
    plt.tight_layout()
    return fig


def plot_geo(geo_zip, state_summary, centroid):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.scatter(geo_zip["lng"], geo_zip["lat"], s=geo_zip["orders"].clip(upper=300) * 0.6 + 2,
               color=ACCENT, alpha=0.30, linewidths=0)
    offsets = {"SP": (-95, -34), "RJ": (30, -22), "MG": (28, 26), "PR": (-95, -4), "RS": (30, -22)}
    for st_code in state_summary.head(5).index:
        if st_code not in centroid.index:
            continue
        ax.annotate(f"{st_code} ({state_summary.loc[st_code, 'revenue_share_pct']:.1f}%)",
                    (centroid.loc[st_code, "geolocation_lng"], centroid.loc[st_code, "geolocation_lat"]),
                    fontsize=10, fontweight="bold", color=WARN, xytext=offsets.get(st_code, (20, 20)),
                    textcoords="offset points", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=WARN, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color=WARN, linewidth=1))
    ax.set_title("Sebaran lokasi pelanggan (ukuran titik = jumlah pesanan)", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    ax = axes[1]
    top = state_summary.head(10).sort_values("revenue")
    hl = min(3, len(top))
    bars = ax.barh(top.index, top["revenue"], color=[MUTED] * (len(top) - hl) + [ACCENT] * hl)
    ax.bar_label(bars, labels=[f"{fmt_brl(r)}  ({p:.1f}%)" for r, p in zip(top["revenue"], top["revenue_share_pct"])],
                 padding=4, fontsize=9)
    ax.set_title("10 negara bagian dengan revenue tertinggi", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total revenue (R$)")
    ax.set_xlim(0, top["revenue"].max() * 1.30)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(fmt_axis))
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Aplikasi
# ----------------------------------------------------------------------------
all_df = load_data()

st.title("Dashboard Analisis E-Commerce Olist")
st.caption("Pesanan berstatus *delivered*, Januari 2017 - Agustus 2018. Revenue = jumlah harga produk (tanpa ongkir), dalam Brazilian Real (R$).")

# ---- Sidebar: filter interaktif
st.sidebar.header("Filter Data")
min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()
date_range = st.sidebar.date_input("Rentang tanggal pesanan", value=(min_date, max_date), min_value=min_date, max_value=max_date)
states = sorted(all_df["customer_state"].unique())
selected_states = st.sidebar.multiselect("Negara bagian pelanggan", options=states, default=states)

if isinstance(date_range, (tuple, list)):
    if len(date_range) != 2:
        st.info("Pilih tanggal akhir pada filter rentang tanggal untuk memperbarui dashboard.")
        st.stop()
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

mask = (
    (all_df["order_purchase_timestamp"].dt.date >= start_date)
    & (all_df["order_purchase_timestamp"].dt.date <= end_date)
    & (all_df["customer_state"].isin(selected_states))
)
df = all_df[mask]

if df.empty:
    st.warning("Tidak ada data untuk filter yang dipilih. Ubah rentang tanggal atau negara bagian.")
    st.stop()

# ---- KPI
total_revenue = df["price"].sum()
total_orders = df["order_id"].nunique()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", fmt_brl(total_revenue))
c2.metric("Jumlah Pesanan", f"{total_orders:,}")
c3.metric("Pelanggan Unik", f"{df['customer_unique_id'].nunique():,}")
c4.metric("Rata-rata Nilai Pesanan", f"R$ {total_revenue / total_orders:,.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["Kategori Produk", "Tren Bulanan", "Segmentasi RFM", "Geospasial"])

# ---- Tab 1: Pertanyaan 1
with tab1:
    st.subheader("Kategori produk apa yang menyumbang revenue tertinggi dan terendah?")
    category_summary = (
        df.groupby("product_category")
        .agg(revenue=("price", "sum"), total_orders=("order_id", "nunique"))
        .sort_values("revenue", ascending=False)
    )
    category_summary["revenue_share_pct"] = category_summary["revenue"] / category_summary["revenue"].sum() * 100
    max_n = int(min(15, max(3, len(category_summary) // 2)))
    n = st.slider("Jumlah kategori yang ditampilkan", min_value=3, max_value=max_n, value=min(10, max_n))
    top_share = category_summary["revenue_share_pct"].head(n).sum()
    st.markdown(f"**{n} kategori teratas menyumbang {top_share:.1f}% dari total revenue** pada filter yang dipilih "
                f"(dari {len(category_summary)} kategori).")
    fig = plot_categories(category_summary, n)
    st.pyplot(fig)
    plt.close(fig)

# ---- Tab 2: Pertanyaan 2
with tab2:
    st.subheader("Bagaimana tren pesanan, revenue, dan pertumbuhan MoM?")
    monthly = (
        df.groupby("month")
        .agg(total_orders=("order_id", "nunique"), revenue=("price", "sum"))
        .sort_index()
    )
    monthly["revenue_mom_pct"] = monthly["revenue"].pct_change() * 100
    peak_month = monthly["revenue"].idxmax()
    st.markdown(f"**Puncak revenue pada {peak_month.strftime('%B %Y')}**: {fmt_brl(monthly.loc[peak_month, 'revenue'])} "
                f"dari {monthly.loc[peak_month, 'total_orders']:,} pesanan.")
    if len(monthly) < 3:
        st.info("Pilih rentang tanggal yang mencakup lebih banyak bulan agar tren dapat terlihat.")
    fig = plot_trend(monthly)
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("Lihat tabel data bulanan"):
        table = monthly.copy()
        table.index = table.index.astype(str)
        st.dataframe(table.rename(columns={"total_orders": "Pesanan", "revenue": "Revenue (R$)", "revenue_mom_pct": "MoM (%)"}).round(2))

# ---- Tab 3: RFM
with tab3:
    st.subheader("Segmentasi pelanggan berdasarkan RFM")
    st.markdown(
        "*Recency* = hari sejak pembelian terakhir, *Frequency* = jumlah pesanan, *Monetary* = total belanja. "
        "Skor Recency dan Monetary dibuat dengan *binning* kuantil (1-5), Frequency dengan batas manual (1, 2, 3+ pesanan)."
    )
    if df["customer_unique_id"].nunique() < 5:
        st.info("Jumlah pelanggan pada filter terlalu sedikit untuk membuat segmentasi RFM.")
    else:
        rfm, reference_date = build_rfm(df)
        seg = (
            rfm.groupby("segment")
            .agg(customers=("customer_unique_id", "count"), avg_recency_days=("recency", "mean"),
                 avg_frequency=("frequency", "mean"), avg_monetary=("monetary", "mean"), total_monetary=("monetary", "sum"))
            .sort_values("customers", ascending=False)
        )
        seg["customer_share_pct"] = seg["customers"] / seg["customers"].sum() * 100
        seg["revenue_share_pct"] = seg["total_monetary"] / seg["total_monetary"].sum() * 100
        one_time = (rfm["frequency"] == 1).mean() * 100
        risk_cust = seg.reindex(["New High-Value", "At-Risk High-Value"]).fillna(0)
        st.markdown(
            f"**{one_time:.1f}% pelanggan hanya berbelanja sekali.** Segmen *New High-Value* dan *At-Risk High-Value* "
            f"bersama-sama {risk_cust['customer_share_pct'].sum():.1f}% pelanggan dan menyumbang "
            f"{risk_cust['revenue_share_pct'].sum():.1f}% revenue. Tanggal acuan: {reference_date:%d %b %Y}."
        )
        fig = plot_rfm(seg)
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("Lihat ringkasan tiap segmen"):
            st.dataframe(seg[["customers", "customer_share_pct", "revenue_share_pct", "avg_recency_days", "avg_frequency", "avg_monetary"]].round(2))

# ---- Tab 4: Geospasial
with tab4:
    st.subheader("Di mana pelanggan dan revenue terkonsentrasi?")
    state_summary = (
        df.groupby("customer_state")
        .agg(revenue=("price", "sum"), total_orders=("order_id", "nunique"))
        .sort_values("revenue", ascending=False)
    )
    state_summary["revenue_share_pct"] = state_summary["revenue"] / state_summary["revenue"].sum() * 100
    geo_df = df.dropna(subset=["geolocation_lat", "geolocation_lng"])
    if geo_df.empty:
        st.info("Tidak ada data koordinat untuk filter yang dipilih.")
    else:
        geo_zip = (
            geo_df.groupby("customer_zip_code_prefix")
            .agg(lat=("geolocation_lat", "first"), lng=("geolocation_lng", "first"),
                 orders=("order_id", "nunique"), revenue=("price", "sum"))
            .reset_index()
        )
        centroid = geo_df.groupby("customer_state")[["geolocation_lat", "geolocation_lng"]].median()
        top3 = state_summary.head(3)
        st.markdown(f"**{', '.join(top3.index)} menyumbang {top3['revenue_share_pct'].sum():.1f}% revenue** pada filter yang dipilih.")
        fig = plot_geo(geo_zip, state_summary, centroid)
        st.pyplot(fig)
        plt.close(fig)

st.caption("Sumber data: Brazilian E-Commerce Public Dataset by Olist. Dibuat sebagai bagian dari proyek akhir analisis data.")
