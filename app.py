import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="E-commerce Product Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("🛍️ E-commerce Product Recommendation using K-Means Clustering")

# Load dataset
df = pd.read_csv("synthetic_ecommerce.csv")

# Aggregate user features
agg = df.groupby(['user_id', 'user_name']).agg(
    total_spend=('total_price', 'sum'),
    avg_price=('price', 'mean'),
    budget=('budget', 'first'),
    num_products=('product_id', 'nunique')
).reset_index()

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(agg[['total_spend', 'avg_price', 'budget', 'num_products']])

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
agg['cluster'] = kmeans.fit_predict(scaled_features)

# Sidebar for user selection
user_list = agg['user_name'].sort_values().unique().tolist()
selected_user = st.sidebar.selectbox("Select User", user_list)

# Show cluster information
user_cluster = agg[agg.user_name == selected_user]['cluster'].values[0]
st.sidebar.markdown("### Cluster Info")
st.sidebar.write(f"User belongs to Cluster: **{user_cluster}**")

# Displaying cluster statistics
st.subheader("📊 Cluster Overview")
cluster_stats = agg.groupby('cluster').agg(
    total_spend_range=('total_spend', lambda x: f"₹{x.min()} - ₹{x.max()}"),
    avg_spend=('total_spend', 'mean'),
    product_count_range=('num_products', lambda x: f"{x.min()} - {x.max()}"),
    avg_budget=('budget', 'mean')
).reset_index()

st.write(cluster_stats)

# Recommendation function (updated default top_n to 5)
def recommend(user_name, top_n=5):
    try:
        cluster = agg[agg.user_name == user_name]['cluster'].values[0]
        cluster_users = agg[agg.cluster == cluster]['user_id']
        filtered_df = df[df.user_id.isin(cluster_users)]
        top_products = (
            filtered_df.groupby(['product_id', 'product_name', 'category', 'price'])
            .size()
            .reset_index(name='count')
            .sort_values(by='count', ascending=False)
            .head(top_n)
        )
        return top_products[['product_id', 'product_name', 'category', 'price']]
    except IndexError:
        return pd.DataFrame(columns=['product_id', 'product_name', 'category', 'price'])

# Display Recommendations with top 5
st.subheader(f"📦 Top Product Recommendations for {selected_user}")
recommended = recommend(selected_user, top_n=5)

if not recommended.empty:
    st.table(recommended.rename(columns={
        'product_id': 'Product ID',
        'product_name': 'Name',
        'category': 'Category',
        'price': 'Price (₹)'
    }))
else:
    st.warning("No recommendations found for this user.")



# Display user's total spend and budget
user_data = agg[agg.user_name == selected_user].iloc[0]
st.subheader(f"💰 Financial Overview for {selected_user}")
st.write(f"**Total Spend:** ₹{user_data['total_spend']}")
st.write(f"**Budget:** ₹{user_data['budget']}")
