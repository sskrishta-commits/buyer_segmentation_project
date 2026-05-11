import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# PAGE CONFIG

st.set_page_config(
    page_title="Real Estate Market Intelligence Dashboard",
    page_icon="🏡",
    layout="wide"
)

# CUSTOM CSS

st.markdown("""
<style>

.main {
    background-color: #f4f6f9;
}

h1 {
    color: #0b1f3a;
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

h2, h3 {
    color: #16324f;
}

div[data-testid="metric-container"] {
    background-color: white;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
    text-align: center;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# TITLE

st.title("🏡 Real Estate Market Intelligence Dashboard")

st.markdown("""
### Machine Learning Based Buyer Segmentation and Investment Profiling

This dashboard uses **Machine Learning (KMeans Clustering)** to analyze  
buyer behavior, investment trends, geographic patterns, and customer segmentation  
within the real estate market.
""")

# LOAD DATA

df = pd.read_csv("final_clustered_data.csv")

# HUMAN READABLE LABELS

country_map = {
    0: "India",
    1: "USA",
    2: "UAE",
    3: "UK",
    4: "Canada"
}

region_map = {
    0: "Asia",
    1: "North America",
    2: "Middle East",
    3: "Europe"
}

client_type_map = {
    0: "Individual",
    1: "Corporate"
}

purpose_map = {
    0: "Investment",
    1: "Personal Use"
}

# MAP VALUES

df['country_name'] = df['country'].map(country_map)

df['region_name'] = df['region'].map(region_map)

df['client_type_name'] = df['client_type'].map(client_type_map)

df['purpose_name'] = df['acquisition_purpose'].map(purpose_map)

# SIDEBAR

st.sidebar.title("🔍 Dashboard Filters")

country = st.sidebar.selectbox(
    "Select Country",
    df['country_name'].dropna().unique()
)

region = st.sidebar.selectbox(
    "Select Region",
    df['region_name'].dropna().unique()
)

client_type = st.sidebar.selectbox(
    "Select Client Type",
    df['client_type_name'].dropna().unique()
)

purpose = st.sidebar.selectbox(
    "Select Acquisition Purpose",
    df['purpose_name'].dropna().unique()
)

segment = st.sidebar.selectbox(
    "Select Buyer Segment",
    df['Buyer_Segment'].unique()
)

# FILTER DATA

filtered_data = df[
    (df['country_name'] == country) &
    (df['region_name'] == region) &
    (df['client_type_name'] == client_type) &
    (df['purpose_name'] == purpose) &
    (df['Buyer_Segment'] == segment)
]

# METRICS SECTION

st.subheader("📊 Key Market Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Buyers",
    len(df)
)

col2.metric(
    "Average Budget",
    f"${df['budget'].mean():,.0f}"
)

col3.metric(
    "Average Satisfaction",
    f"{df['satisfaction_score'].mean():.1f}"
)

col4.metric(
    "Buyer Segments",
    df['Buyer_Segment'].nunique()
)

# DATA PREVIEW

st.subheader("📁 Filtered Buyer Data")

st.dataframe(
    filtered_data,
    use_container_width=True
)

# FIRST ROW

col5, col6 = st.columns(2)

with col5:

    st.subheader("🥧 Buyer Segment Distribution")

    fig1, ax1 = plt.subplots(figsize=(6,6))

    df['Buyer_Segment'].value_counts().plot(
        kind='pie',
        autopct='%1.1f%%',
        ax=ax1
    )

    ax1.set_ylabel("")

    st.pyplot(fig1)

with col6:

    st.subheader("🌍 Geographic Buyer Analysis")

    fig2, ax2 = plt.subplots(figsize=(7,4))

    df['country_name'].value_counts().plot(
        kind='bar',
        ax=ax2
    )

    ax2.set_xlabel("Country")
    ax2.set_ylabel("Number of Buyers")

    st.pyplot(fig2)

# SECOND ROW

col7, col8 = st.columns(2)

with col7:

    st.subheader("💰 Investor Behaviour Analysis")

    fig3, ax3 = plt.subplots(figsize=(7,4))

    ax3.scatter(
        df['age'],
        df['budget'],
        c=df['Cluster']
    )

    ax3.set_xlabel("Age")
    ax3.set_ylabel("Budget")

    st.pyplot(fig3)

with col8:

    st.subheader("⭐ Satisfaction Score Distribution")

    fig4, ax4 = plt.subplots(figsize=(7,4))

    ax4.hist(df['satisfaction_score'])

    ax4.set_xlabel("Satisfaction Score")
    ax4.set_ylabel("Count")

    st.pyplot(fig4)

# HEATMAP

st.subheader("🔥 Correlation Heatmap")

numeric_df = df.select_dtypes(include=['number'])

correlation = numeric_df.corr()

fig5, ax5 = plt.subplots(figsize=(10,6))

heatmap = ax5.imshow(correlation)

ax5.set_xticks(range(len(correlation.columns)))
ax5.set_yticks(range(len(correlation.columns)))

ax5.set_xticklabels(
    correlation.columns,
    rotation=90
)

ax5.set_yticklabels(
    correlation.columns
)

plt.colorbar(heatmap)

st.pyplot(fig5)

# CLIENT TYPE PIE CHART

st.subheader("👥 Client Type Distribution")

fig6, ax6 = plt.subplots(figsize=(6,6))

df['client_type_name'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    ax=ax6
)

ax6.set_ylabel("")

st.pyplot(fig6)

# SEGMENT INSIGHTS PANEL

st.subheader("🧠 Segment Insights Panel")

cluster_summary = df.groupby(
    'Buyer_Segment'
).mean(numeric_only=True)

st.dataframe(
    cluster_summary,
    use_container_width=True
)

# PROJECT INSIGHTS

st.subheader("📌 Key Business Insights")

st.markdown("""
- Luxury Investors generally show higher budgets and investment capacity.
- First-Time Buyers prefer lower budget properties with shorter investment horizons.
- Corporate Buyers contribute significantly to premium property investments.
- Geographic analysis helps identify region-wise investment demand.
- Satisfaction analysis helps improve customer targeting and marketing strategy.
""")

# TECHNOLOGIES USED

st.subheader("⚙️ Technologies Used")

st.markdown("""
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-Learn  
- Streamlit  
- KMeans Clustering  
""")

# FOOTER

st.markdown("---")

st.markdown("""
### 👨‍💻 Academic Project Submission

Machine Learning Based Buyer Segmentation and Investment Profiling  
for Real Estate Market Intelligence
""")
