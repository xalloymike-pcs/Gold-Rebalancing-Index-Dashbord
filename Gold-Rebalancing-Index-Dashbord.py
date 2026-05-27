import numpy as np
import pandas as pd
import yfinance as yf
import altair as alt
import streamlit as st

# =========================
# 頁面設定
# =========================

st.set_page_config(
    page_title="Gold Rebalancing Index Dashbord",
    layout="wide"
)

st.markdown("""
<style>

/* 主背景 */
.stApp {
    background-color: #0B0F17;
    color: #FAFAFA;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #131A24;
}

/* metric 卡片 */
[data-testid="metric-container"] {
    background-color: #131A24;
    border: 1px solid #222;
    padding: 12px;
    border-radius: 12px;
}

/* DataFrame */
[data-testid="stDataFrame"] {
    background-color: #131A24;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: white;
}

/* 隱藏 Streamlit Header */
header[data-testid="stHeader"]{
    display: none;
}

/* 頁面往上重排 */
.main .block-container{
    padding-top: 1rem;
}

/* Title 修正 */
h1{
    margin-top: 0 !important;
    padding-top: 0 !important;
}

</style>
""", unsafe_allow_html=True)

st.title("📊 Gold Rebalancing Index Dashboard")
st.markdown(
    """
    <div style="
        font-size:22px;
        color:#D4AF37;
        padding:8px 12px;
        border-left:3px solid #D4AF37;
        background-color:rgba(255,255,255,0.03);
        border-radius:6px;
        margin-bottom:15px;
    ">
    ⚠️ Data sourced from Yahoo Finance continuous front-month gold futures contracts. 
    Intended for visualization and volatility research only, not for execution or precise trading decisions.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='font-size:22px; color:white;'>Core：Volatility、Trend、Band、Regime、Suggestion</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='font-size:22px; color:white;'>Extra：ATR、ADX、Momentum、BB Width、RSI</p>",
    unsafe_allow_html=True
)

# =========================
# 📥 Download Yahoo Data
# =========================

df = yf.download(
    "GC=F",
    period="3y",
    interval="1d",
    auto_adjust=True
)
# =========================
# 修正欄位
# =========================

# MultiIndex 修正
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# index 轉欄位
df = df.reset_index()

# 日期欄位自動辨識
date_col = df.columns[0]

# =========================
# 重新命名
# =========================

df = df.rename(columns={
    date_col: "Date"
})

# =========================
# 只保留需要資料
# =========================

cols = ["Date", "High", "Low"]

if "Close" in df.columns:
    cols.append("Close")

elif "Adj Close" in df.columns:
    df["Close"] = df["Adj Close"]
    cols.append("Close")

else:
    st.error("找不到 Close 欄位")
    st.stop()

df = df.loc[:, cols]

# =========================
# 資料清洗
# =========================

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=["High", "Low", "Close"])

st.write("有效資料筆數:", len(df))

if len(df) < 20:
    st.error("資料不足，無法計算策略")
    st.stop()

# =========================
# 拆資料
# =========================

history = list(df.itertuples(index=False, name=None))

dates = [x[0] for x in history]

high = np.array([x[1] for x in history])
low = np.array([x[2] for x in history])
close = np.array([x[3] for x in history])

latest_date = dates[-1]
latest_close = close[-1]

# =========================
# 📊 1. Volatility (Parkinson)、波動率
# =========================

rs = np.log(high / low) ** 2

vol_raw = np.sqrt(rs / (4 * np.log(2)))
vol_smooth = pd.Series(vol_raw).ewm(span=14).mean()

vol = vol_smooth.iloc[-1] * np.sqrt(252)

# =========================
# 📈 2. Trend 趨勢
# =========================

trend_series = pd.Series(close).pct_change(5)
trend = trend_series.iloc[-1]

# =========================
# 📉 3. Band deviation 布林帶
# =========================

s = pd.Series(close)

ma = s.ewm(span=10).mean()
std = s.ewm(span=10).std()

latest_price = s.iloc[-1]
latest_ma = ma.iloc[-1]
latest_std = max(std.iloc[-1], 1e-6)

band_dev = (latest_price - latest_ma) / latest_std

# =========================
# 🧠4. Regime、市場狀態（用歷史資料）
# =========================

vol_series = pd.Series(np.sqrt(rs))

trend_series = pd.Series(close).pct_change(5)

vol_mean = vol_series.ewm(span=5).mean()
vol_std = vol_series.ewm(span=5).std()
vol_z = (vol_series - vol_mean) / vol_std

trend_mean = trend_series.ewm(span=5).mean()
trend_std = trend_series.ewm(span=5).std()
trend_z = (trend_series - trend_mean) / trend_std

vol_z_latest = vol_z.iloc[-1]
trend_z_latest = trend_z.iloc[-1]

def get_regime(vol_z, trend_z):

    if abs(vol_z) > 0.5 and abs(trend_z) < 0.2:
        return "Mean Reversion、均值回歸"

    elif abs(trend_z) > 0.5:
        return "Trend、趨勢延續"

    else:
        return "Mixed、不明確"

regime = get_regime(vol_z_latest, trend_z_latest)

# =========================
# 🎯 5. Last suggestion、最新建議
# =========================

def get_action(regime, band_dev, trend):

    if "Trend、趨勢延續" in regime:
        if trend > 0:
            return "做多、順勢"
        else:
            return "做空、順勢"

    if "Mean Reversion、均值回歸" in regime:

        if abs(trend) > 0.5:
            return "小倉、觀望"

        if band_dev > 2:
            return "做空、過熱回檔"

        elif band_dev < -2:
            return "做多、超跌反彈"

        else:
            return "小倉、觀望"

    return "小倉、觀望"

action = get_action(regime, band_dev, trend)

# =========================
# ATR、市場真實波動：高=大行情、低=盤整
# =========================

n = 14

df2 = pd.DataFrame({"high": high,"low": low,"close": close})

prev_close = df2["close"].shift(1)

tr = pd.DataFrame({
    "hl": df2["high"] - df2["low"],
    "hc": (df2["high"] - prev_close).abs(),
    "lc": (df2["low"] - prev_close).abs()
}).max(axis=1)

tr.iloc[0] = df2["high"].iloc[0] - df2["low"].iloc[0]

atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

atr_value = atr.iloc[-1]

atr_norm = atr / df2["close"]
atr_value_norm = atr_norm.iloc[-1]

# =========================
# Bollinger width、壓縮或擴張：壓縮=市場安靜(<0.03)、寬=市場劇烈(>0.1)
# =========================

bb_width_series = (std / ma).ewm(span=10).mean()
bb_width = bb_width_series.iloc[-1]

# =========================
# Momentum、趨勢速度：大於0上升動能、小於0下跌動能
# =========================

ret = close[-1] / close[-5] - 1
momentum_score = ret / (vol + 1e-6)

# =========================
# RSI、買賣超：大於70超買、小於30超賣
# =========================

delta = pd.Series(close).diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()

rs = avg_gain / (avg_loss + 1e-9)
rsi = 100 - (100 / (1 + rs))

rsi_value = rsi.iloc[-1]

# =========================
# ADX（簡化）、趨勢強度：大於25有趨勢、小於20盤整
# =========================

n = 14

df = pd.DataFrame({"high": high, "low": low, "close": close})

# === True Range ===
prev_close = df["close"].shift(1)

tr = pd.DataFrame({
    "hl": df["high"] - df["low"],
    "hc": (df["high"] - prev_close).abs(),
    "lc": (df["low"] - prev_close).abs()
}).max(axis=1)

tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]

atr = tr.ewm(alpha=1/n, adjust=False).mean()

# === Directional Movement ===
up_move = df["high"].diff()
down_move = -df["low"].diff()

plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

plus_dm = pd.Series(plus_dm, index=df.index)
minus_dm = pd.Series(minus_dm, index=df.index)

# === Wilder smoothing ===
plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr
minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr

# === DX ===
dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100

# === ADX ===
adx = dx.ewm(alpha=1/n, adjust=False).mean()

adx_value = adx.iloc[-1]

# =========================
# Dashboard
# =========================

st.subheader(f"📅 Latest Date : {latest_date}")

# =========================
# Price Chart
# =========================
df = pd.DataFrame({ "Date": dates,"High": high, "Low": low, "Close": close})


# 長格式資料
chart_df = df.melt(
    id_vars=["Date"],
    value_vars=["High", "Low", "Close"],
    var_name="Type",
    value_name="Price"
)

# 滑鼠最近點選擇
nearest = alt.selection_point(
    nearest=True,
    on="pointermove",
    fields=["Date"],
    empty=False
)

# 基礎線圖
line = (
    alt.Chart(chart_df)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y( "Price:Q", title="Price", scale=alt.Scale(zero=False)),
        color=alt.Color( "Type:N", scale=alt.Scale( domain=["High", "Low", "Close"], range=["red", "#66FF66", "gold"])
        )
    )
)

# 滑鼠感應透明層
selectors = (
    alt.Chart(df)
    .mark_point(opacity=0)
    .encode(
        x="Date:T"
    )
    .add_params(nearest)
)

# 垂直線
rules = (
    alt.Chart(df)
    .mark_rule(color="gray")
    .encode(
        x="Date:T"
    )
    .transform_filter(nearest)
)

# 三個價格圓點
points = (
    line.mark_circle(size=90)
    .encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
)

# 固定資訊
info_df = pd.DataFrame({ "x": [10], "y": [10]})

text_date = (
    alt.Chart(df)
    .mark_text(
        align="left",
        fontSize=22,
        fontWeight="bold",
	color="white", 
        dx=10,
        dy=10
    )
    .encode(
        text=alt.condition(
            nearest,
            alt.Text("Date:T", format="%Y-%m-%d"),
            alt.value("")
        )
    )
    .transform_filter(nearest)
)

text_high = (
    alt.Chart(df)
    .mark_text(
        align="left",
        color="red",
        fontSize=22,
        dx=10,
        dy=35
    )
    .encode(
        text=alt.condition(
            nearest,
            alt.Text("High:Q", format=".2f"),
            alt.value("")
        )
    )
    .transform_filter(nearest)
)

text_low = (
    alt.Chart(df)
    .mark_text(
        align="left",
        color="#66FF66",
        fontSize=22,
        dx=10,
        dy=55
    )
    .encode(
        text=alt.condition(
            nearest,
            alt.Text("Low:Q", format=".2f"),
            alt.value("")
        )
    )
    .transform_filter(nearest)
)

text_close = (
    alt.Chart(df)
    .mark_text(
        align="left",
        color="goldenrod",
        fontSize=22,
        dx=10,
        dy=75
    )
    .encode(
        text=alt.condition(
            nearest,
            alt.Text("Close:Q", format=".2f"),
            alt.value("")
        )
    )
    .transform_filter(nearest)
)

# 合併圖表
chart = (
    alt.layer(
        line,
        selectors,
        rules,
        points,
        text_date,
        text_high,
        text_low,
        text_close
    )
    .properties(
        width=1100,
        height=500
    )
    .configure(
        background="#0B0F17"
    )
    .configure_view(
        fill="#0B0F17",
        stroke=None
    )
    .configure_axis(
        labelColor="#D1D4DC",
        titleColor="#D1D4DC",
        gridColor="#1F2937",
        domainColor="#374151",
        tickColor="#374151",
        labelFontSize=13,
        titleFontSize=14
    )
    .configure_legend(
        labelColor="#D1D4DC",
        titleColor="#D1D4DC",
        orient="top"
    )
    .interactive()
)

# Streamlit 顯示
st.altair_chart(chart, use_container_width=True)

# =========================
# Professional UI
# =========================

# =========================
# Core Indicators
# =========================

st.subheader("📊 Core Indicators")

st.markdown("""
<style>

/* 卡片外框 */
.block-container {
    padding-top: 1rem;
}

/* 主標題數字 */
.big-number {
    font-size: 40px;
    font-weight: 700;
    color: white;
    line-height: 1.2;
}

/* 標題 */
.title {
    font-size: 22px;
    font-weight: 700;
    color: #E0E0E0;
}

/* 副標 */
.subtitle {
    font-size: 22px;
    color: #A0A0A0;
}

/* card spacing */
.card {
    padding: 8px 4px;
}

</style>
""", unsafe_allow_html=True)

#第一欄
col1, col2, col3, col4, col5 = st.columns(5)

# Volatility
with col1:
    st.markdown("""
    <div class="card">
        <div class="title">Volatility、年化波動率</div>
        <div class="subtitle">最近14天；2025年約14-22%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='big-number'>{vol:.2%}</div>", unsafe_allow_html=True)

# Trend
with col2:
    st.markdown("""
    <div class="card">
        <div class="title">Trend、趨勢</div>
        <div class="subtitle">正常|trend|< 0.02，超過有波動</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='big-number'>{trend:.4f}</div>", unsafe_allow_html=True)

# Band Deviation
with col3:
    st.markdown("""
    <div class="card">
        <div class="title">Band Deviation、布林帶</div>
        <div class="subtitle">正常|dev|< 0.5，超過有波動</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='big-number'>{band_dev:.4f}</div>", unsafe_allow_html=True)

# Market Regime
with col4:
    st.markdown("""
    <div class="card">
        <div class="title">Market Regime、市場狀態</div>
        <div class="subtitle">均值回歸、趨勢延續、不明確</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='big-number'>{regime}</div>", unsafe_allow_html=True)

# Last Suggestion
with col5:
    st.markdown("""
    <div class="card">
        <div class="title">Last Suggestion、最新建議</div>
        <div class="subtitle">123456789</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='big-number'>{action}</div>", unsafe_allow_html=True)


# =========================
# Extra Indicators
# =========================

st.subheader("📊 Extra Indicators")

st.markdown("""
<style>
.metric-card{
    background: linear-gradient(145deg,#111827,#1f2937);
    padding: 20px 22px;

    /* 卡片大小 */
    min-height: 210px;   /* 原本165px 太小 */
    height: auto;        /* 讓內容自動撐開 */

    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);

    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    transition: all 0.2s ease;

    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.metric-card:hover{
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}

.metric-title{
    font-size: 22px;
    color: #9ca3af;
    font-weight: 600;
    margin-bottom: 10px;
}

.metric-value{
    font-size: 40px;
    font-weight: 700;
    color: white;
    margin-bottom: 12px;
}

.metric-desc{
    font-size: 22px;
    color: #d1d5db;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


#第二欄

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">ATR、原始波動</div>
        <div class="metric-value">{atr_value:.4f}</div>
        <div class="metric-desc">
            平均波動幅度<br>
            數值越高波動越劇烈
        </div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">NATR、標準化波動</div>
        <div class="metric-value">{atr_value_norm:.2f}</div>
        <div class="metric-desc">
            價格百分比波動<br>
            可跨商品比較波動性
        </div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">RSI、相對強弱</div>
        <div class="metric-value">{rsi_value:.2f}</div>
        <div class="metric-desc">
            RSI &lt; 30：超賣<br>
            RSI &gt; 70：超買
        </div>
    </div>
    """, unsafe_allow_html=True)


#第三欄

col7, col8, col9 = st.columns(3)

with col7:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">ADX、趨勢強度</div>
        <div class="metric-value">{adx_value:.2f}</div>
        <div class="metric-desc">
            ADX &lt; 20：盤整<br>
            ADX &gt; 25：趨勢行情
        </div>
    </div>
    """, unsafe_allow_html=True)

with col8:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Momentum、趨勢動能</div>
        <div class="metric-value">{momentum_score:.2f}</div>
        <div class="metric-desc">
            Momentum &lt; 0：下跌動能<br>
            Momentum &gt; 0：上升動能
        </div>
    </div>
    """, unsafe_allow_html=True)

with col9:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">BB Width、市場環境</div>
        <div class="metric-value">{bb_width:.4f}</div>
        <div class="metric-desc">
            &lt; 0.03：安靜<br>
            &gt; 0.08：吵鬧
        </div>
    </div>
    """, unsafe_allow_html=True)

