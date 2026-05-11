import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Real Estate Buyer Segmentation Dashboard")

# LOAD DATA

df = pd.read_csv("final_clustered_data.csv")

# SHOW DATA

st.subheader("Dataset Preview")

st.write(df.head())

# FILTER

segment = st.selectbox(
    "Select Buyer Segment",
    df['Buyer_Segment'].unique()
)

filtered_data = df[
    df['Buyer_Segment'] == segment
]

st.subheader("Filtered Buyer Data")

st.write(filtered_data)

# BAR GRAPH

st.subheader("Buyer Segment Distribution")

fig, ax = plt.subplots()

df['Buyer_Segment'].value_counts().plot(
    kind='bar',
    ax=ax
)

plt.xlabel("Buyer Segment")

plt.ylabel("Count")

st.pyplot(fig)

# SCATTER GRAPH

st.subheader("Age vs Budget")

fig2, ax2 = plt.subplots()

scatter = ax2.scatter(
    df['age'],
    df['budget'],
    c=df['Cluster']
)

ax2.set_xlabel("Age")

ax2.set_ylabel("Budget")

st.pyplot(fig2)

# HISTOGRAM

st.subheader("Satisfaction Score Distribution")

fig3, ax3 = plt.subplots()

ax3.hist(df['satisfaction_score'])

st.pyplot(fig3)
