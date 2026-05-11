import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# PAGE TITLE

st.title("Machine Learning Based Buyer Segmentation and Investment Profiling")

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

# CONVERT ENCODED VALUES TO TEXT

df['country_name'] = df['country'].map(country_map)

df['region_name'] = df['region'].map(region_map)

df['client_type_name'] = df['client_type'].map(client_type_map)

df['purpose_name'] = df['acquisition_purpose'].map(purpose_map)

# DATASET PREVIEW

st.subheader("Dataset Preview")

st.write(df.head())

# SIDEBAR FILTERS

st.sidebar.header("User Controls / Filters")

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

ax2.scatter(
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
ax3.set_title("Satisfaction Score Distribution")

st.pyplot(fig3)

# GEOGRAPHIC BUYER ANALYSIS

st.subheader("Geographic Buyer Analysis")

country_counts = df['country_name'].value_counts()

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
).mean(numeric_only=True)

st.write(cluster_summary)
