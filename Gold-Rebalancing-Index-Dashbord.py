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
	margin-top:15px;
        margin-bottom:15px;
    ">
    ⚠️ Data sourced from Yahoo Finance continuous front-month gold futures contracts. 
    Intended for visualization and volatility research only, not for execution or precise trading decisions.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='font-size:22px; color:white;'>Core：Volatility、Trend、Band Deviation、Regime、Suggestion</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='font-size:22px; color:white;'>Extra：ATR、NATR、RSI、ADX、Momentum、Bollinger Band Width</p>",
    unsafe_allow_html=True
)

# =========================
# 📥 Download Yahoo Data
# =========================

df = yf.download(
    "GC=F",
    period="2y",
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

# Data Validation
valid_rows = len(df)
if "Date" in df.columns:
    start_date = df["Date"].iloc[0].strftime("%Y-%m-%d")
    end_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")
else:
    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].strftime("%Y-%m-%d")

# Info Bar
col1, col2, col3 = st.columns(3)

st.markdown("""
<style>
/* Metric Label */
[data-testid="stMetricLabel"] {
    color: #888888;
    font-size: 28px;
    font-weight: 600;
}

/* Metric Value */
[data-testid="stMetricValue"] {
    color: #D4AF37;
    font-size: 32px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

col1.metric("有效資料筆數", f"{valid_rows:,}")
col2.metric("起始日期", start_date)
col3.metric("最新日期", end_date)

# Minimum Data Check
if valid_rows < 60:

    st.error(
        "資料不足，至少需要 60 筆資料才能計算 "
        "Volatility / Trend / Regime Strategy"
    )

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

# 通用=========================

high_s = pd.Series(high)
low_s = pd.Series(low)
close_s = pd.Series(close)

ema20 = close_s.ewm(span=20).mean()
ema60 = close_s.ewm(span=60).mean()
std20 = close_s.ewm(span=20).std()

n = 14
lookback = 20

# =========================
# 📊 1. Parkinson Volatility、波動率
# =========================

parkinson_rs = np.log(high_s / low_s) ** 2
parkinson_var = (parkinson_rs/ (4 * np.log(2)))
parkinson_vol_series = (parkinson_var.ewm(span=20).mean())

vol = (np.sqrt(parkinson_vol_series.iloc[-1])* np.sqrt(252))

# =========================
# 📈 2. Trend 趨勢
# =========================

trend_series = np.log(ema20 / ema60)
trend = trend_series.iloc[-1]

# =========================
# 📉 3. Band deviation 布林帶偏離度
# =========================

latest_price = close_s.iloc[-1]
band_dev = (latest_price - ema20.iloc[-1]) / max(std20.iloc[-1], 1e-6)

# =========================
# 🧠4. Regime、市場狀態（用歷史資料）
# =========================

vol_series = np.sqrt(parkinson_vol_series)
vol_mean = vol_series.ewm(span=10).mean()
vol_std = (vol_series.ewm(span=10).std().clip(lower=1e-6))
vol_z = (vol_series - vol_mean) / vol_std
vol_z_latest = vol_z.iloc[-1]

def get_regime(vol_z, trend):

    if abs(vol_z) > 0.5 and abs(trend) < 0.01:
        return "Mean Reversion、均值回歸"

    elif abs(trend) > 0.01:
        return "Trend、趨勢延續"

    else:
        return "Mixed、不明確"

regime = get_regime(vol_z_latest,trend)

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

        if abs(trend) > 0.01:
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

df = pd.DataFrame({"high": high,"low": low,"close": close})
prev_close = df["close"].shift(1)

# === True Range ===
tr = pd.DataFrame({
    "hl": df["high"] - df["low"],
    "hc": (df["high"] - prev_close).abs(),
    "lc": (df["low"] - prev_close).abs()
}).max(axis=1)

tr.iloc[0] = (df["high"].iloc[0]- df["low"].iloc[0])

# === ATR (Wilder) ===
atr = tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
atr_value = atr.iloc[-1]

# === Normalized ATR ===
atr_norm = atr / df["close"]
atr_value_norm = atr_norm.iloc[-1]

# =========================
# Bollinger width、壓縮或擴張：壓縮=市場安靜(<0.05)、寬=市場劇烈(>0.08)
# =========================

ma20 = close_s.rolling(20).mean()
std20 = close_s.rolling(20).std()

bb_width_series = 4 * std20 / ma20
bb_width = bb_width_series.iloc[-1]

bb_percentile = bb_width_series.rank(pct=True).iloc[-1]

# =========================
# Momentum、趨勢速度：大於0上升動能、小於0下跌動能
# =========================

log_ret = np.log(close_s.iloc[-1]/ close_s.iloc[-lookback])
momentum_score = log_ret / (vol + 1e-6)

# =========================
# RSI、買賣超：大於65超買、小於35超賣
# =========================

delta = close_s.diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()

rsi_rs = avg_gain / (avg_loss + 1e-9)
rsi = 100 - (100 / (1 + rsi_rs))
rsi_value = rsi.iloc[-1]

# =========================
# ADX（簡化）、趨勢強度：大於25有趨勢、小於20盤整
# =========================

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
adx = dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

adx_value = adx.iloc[-1]

# =========================
# MA
# =========================
ma5 = close_s.rolling(5).mean()
ma15 = close_s.rolling(15).mean()
ma30 = close_s.rolling(30).mean()
ma60 = close_s.rolling(60).mean()
ma100 = close_s.rolling(100).mean()
ma200 = close_s.rolling(200).mean()

# 最新值
ma5_value = ma5.iloc[-1]
ma15_value = ma15.iloc[-1]
ma30_value = ma30.iloc[-1]
ma60_value = ma60.iloc[-1]
ma100_value = ma100.iloc[-1]
ma200_value = ma200.iloc[-1]

# =========================
# Dashboard
# =========================

st.subheader(f"📈 COMEX Gold Continuous Futures")

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
INFO_X = 20
INFO_Y = 20
LINE_SPACING = 28

text_date = (
    alt.Chart(df)
    .mark_text(
        align="left",
        fontSize=22,
        fontWeight="bold",
        color="white"
    )
    .encode(
   	x=alt.value(INFO_X),
    	y=alt.value(INFO_Y),

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
        fontSize=22
    )
    .encode(
    	x=alt.value(INFO_X),
    	y=alt.value(INFO_Y + LINE_SPACING),

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
        fontSize=22
    )
    .encode(
    	x=alt.value(INFO_X),
    	y=alt.value(INFO_Y + LINE_SPACING * 2),

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
        fontSize=22
    )
    .encode(
    	x=alt.value(INFO_X),
    	y=alt.value(INFO_Y + LINE_SPACING * 3),

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

# Core Indicators====================================

st.subheader("🎯 Core Indicators")

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
        <div class="title">Parkinson Volatility、年化波動率</div>
        <div class="subtitle">20D、過去三年約13%-18%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='big-number'>{vol:.2%}</div>", unsafe_allow_html=True)

# Trend
with col2:
    st.markdown("""
    <div class="card">
        <div class="title">Trend、趨勢</div>
        <div class="subtitle">正常|trend|< 0.03，波動顯著</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='big-number'>{trend:.4f}</div>", unsafe_allow_html=True)

# Band Deviation
with col3:
    st.markdown("""
    <div class="card">
        <div class="title">Band Deviation、布林帶偏離度</div>
        <div class="subtitle">正常|dev|< 1，波動顯著</div>
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
        <div class="subtitle">-</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='big-number'>{action}</div>", unsafe_allow_html=True)


# Extra Indicators====================================

st.subheader("🧭 Extra Indicators")

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
        <div class="metric-value">{atr_value:.2f}</div>
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
        <div class="metric-value">{atr_value_norm:.4f}</div>
        <div class="metric-desc">
            價格百分比波動，正常約0.02，數值越大波動程度越高<br>
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
            RSI &lt; 25：超賣<br>
            RSI &gt; 75：超買
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
            ADX &gt; 25：趨勢較顯著
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
        <div class="metric-title">Bollinger Band Width、布林帶寬度</div>
        <div class="metric-value">{bb_width:.2f}</div>
        <div class="metric-desc">
            Historical Percentile：{bb_percentile:.0%}<br>
            &lt; 0.05：市場情緒安靜<br>
            &gt; 0.08：市場情緒吵鬧
        </div>
    </div>
    """, unsafe_allow_html=True)

#第四、五欄

col10, col11, col12 = st.columns(3)

with col10:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">MA5、5日均線</div>
        <div class="metric-value">{ma5_value:,.2f}</div>
        <div class="metric-desc">Short-Term Trend</div>
    </div>
    """, unsafe_allow_html=True)

with col11:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">MA15、15日均線</div>
        <div class="metric-value">{ma15_value:,.2f}</div>
        <div class="metric-desc">Short-Term Trend</div>
    </div>
    """, unsafe_allow_html=True)

with col12:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">MA30、30日均線</div>
        <div class="metric-value">{ma30_value:,.2f}</div>
        <div class="metric-desc">Medium-Term Trend</div>
    </div>
    """, unsafe_allow_html=True)

col13, col14, col15 = st.columns(3)

with col13:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">MA60、60日均線</div>
        <div class="metric-value">{ma60_value:,.2f}</div>
        <div class="metric-desc">Medium-Term Trend</div>
    </div>
    """, unsafe_allow_html=True)

with col14:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">MA100、100日均線</div>
        <div class="metric-value">{ma100_value:,.2f}</div>
        <div class="metric-desc">Long-Term Trend</div>
    </div>
    """, unsafe_allow_html=True)

with col15:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">MA200、200日均線</div>
        <div class="metric-value">{ma200_value:,.2f}</div>
        <div class="metric-desc">Long-Term Trend</div>
    </div>
    """, unsafe_allow_html=True)

#底部說明

st.markdown(
    """
    <div style="
        font-size:22px;
        color:#D4AF37;
        padding:8px 12px;
        border-left:3px solid #D4AF37;
        background-color:rgba(255,255,255,0.03);
        border-radius:6px;
	margin-top:15px;
        margin-bottom:15px;
    ">
    ⚠️ Differences between indicator signals may arise, as each model captures distinct market structures, time horizons, and behavioral dynamics. This is a natural characteristic of financial markets.
    </div>
    """,
    unsafe_allow_html=True
)

