import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# PAGE TITLE

st.title("Machine Learning Based Buyer Segmentation and Investment Profiling")

# LOAD DATA

df = pd.read_csv("final_clustered_data.csv")

# DATASET PREVIEW

st.subheader("Dataset Preview")

st.write(df.head())

# SIDEBAR FILTERS

st.sidebar.header("User Controls / Filters")

# COUNTRY FILTER

country = st.sidebar.selectbox(
    "Select Country",
    df['country'].unique()
)

# REGION FILTER

region = st.sidebar.selectbox(
    "Select Region",
    df['region'].unique()
)

# CLIENT TYPE FILTER

client_type = st.sidebar.selectbox(
    "Select Client Type",
    df['client_type'].unique()
)

# ACQUISITION PURPOSE FILTER

purpose = st.sidebar.selectbox(
    "Select Acquisition Purpose",
    df['acquisition_purpose'].unique()
)

# BUYER SEGMENT FILTER

segment = st.sidebar.selectbox(
    "Select Buyer Segment",
    df['Buyer_Segment'].unique()
)

# FILTERED DATA

filtered_data = df[
    (df['country'] == country) &
    (df['region'] == region) &
    (df['client_type'] == client_type) &
    (df['acquisition_purpose'] == purpose) &
    (df['Buyer_Segment'] == segment)
]

# SHOW FILTERED DATA

st.subheader("Filtered Buyer Data")

st.write(filtered_data)

# BUYER SEGMENT OVERVIEW

st.subheader("Buyer Segmentation Overview")

fig, ax = plt.subplots()

df['Buyer_Segment'].value_counts().plot(
    kind='bar',
    ax=ax
)

plt.xlabel("Buyer Segment")
plt.ylabel("Count")
plt.title("Buyer Segment Distribution")

st.pyplot(fig)

# INVESTOR BEHAVIOUR DASHBOARD

st.subheader("Investor Behaviour Dashboard")

fig2, ax2 = plt.subplots()

scatter = ax2.scatter(
    df['age'],
    df['budget'],
    c=df['Cluster']
)

ax2.set_xlabel("Age")
ax2.set_ylabel("Budget")
ax2.set_title("Age vs Budget")

st.pyplot(fig2)

# SATISFACTION SCORE ANALYSIS

st.subheader("Satisfaction Score Analysis")

fig3, ax3 = plt.subplots()

ax3.hist(df['satisfaction_score'])

ax3.set_xlabel("Satisfaction Score")
ax3.set_ylabel("Count")

st.pyplot(fig3)

# GEOGRAPHIC BUYER ANALYSIS

st.subheader("Geographic Buyer Analysis")

country_counts = df['country'].value_counts()

fig4, ax4 = plt.subplots()

country_counts.plot(
    kind='bar',
    ax=ax4
)

plt.xlabel("Country")
plt.ylabel("Number of Buyers")
plt.title("Country Wise Buyer Analysis")

st.pyplot(fig4)

# SEGMENT INSIGHTS PANEL

st.subheader("Segment Insights Panel")

cluster_summary = df.groupby(
    'Buyer_Segment'
).mean()

st.write(cluster_summary)

