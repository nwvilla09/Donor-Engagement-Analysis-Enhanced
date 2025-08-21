import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the simulated donor data (mirroring CRM export)
df = pd.read_csv('donor_data.csv')

# Data cleaning (ensure integrity, as per job responsibilities)
df['DonationAmount'] = df['DonationAmount'].clip(lower=0)  # No negative donations
df['ResponseRate'] = df['ResponseRate'].clip(0, 1)  # Bound between 0 and 1

# Summary analysis: Aggregate by IncomeLevel for segmentation insights
# Includes averages for donations, interactions, response rates; count of engaged donors
summary = df.groupby('IncomeLevel').agg({
    'DonationAmount': 'mean',
    'NumInteractions': 'mean',
    'ResponseRate': 'mean',
    'Engaged': 'sum'
}).rename(columns={
    'DonationAmount': 'Avg_Donation',
    'NumInteractions': 'Avg_Interactions',
    'ResponseRate': 'Avg_Response_Rate',
    'Engaged': 'Engaged_Count'
})
summary['Total_Donors'] = df.groupby('IncomeLevel').size()
summary['Engagement_Rate'] = summary['Engaged_Count'] / summary['Total_Donors']  # For retention/engagement metrics

# Export summary for reporting (e.g., campaign performance)
summary.to_csv('donor_summary.csv')

# Visualizations for communication strategies and ROI analysis
# Bar chart: Average donations by income level (for targeted appeals)
plt.figure(figsize=(8, 6))  # Larger size for readability
bars = plt.bar(summary.index, summary['Avg_Donation'], color=['#2ECC71', '#27AE60', '#219653'])  # Green shades for nonprofit theme
plt.title('Average Donation by Income Level for Targeted Appeals')
plt.xlabel('Income Level')
plt.ylabel('Average Donation ($)')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)  # Add horizontal grid
# Add data labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f'${yval:.2f}', 
             ha='center', va='bottom', fontsize=10)
# Add annotation for highest-giving group, adjusted to avoid overlap
max_income = summary['Avg_Donation'].idxmax()
max_donation = summary['Avg_Donation'].max()
# Position annotation higher to avoid grid lines, with a white background
plt.text(0.5, max_donation + 50, f'Highest: {max_income} (${max_donation:.2f})', 
         ha='center', va='bottom', fontsize=10, color='#145A32',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.savefig('avg_donation_bar.png')
plt.close()

# Pie chart: Engagement distribution (for stewardship and retention strategies)
engaged_counts = df['Engaged'].value_counts()
plt.figure(figsize=(8, 8))  # Larger size for readability
colors = ['#FF4C4C', '#4C78FF']  # Red for Not Engaged, Blue for Engaged
plt.pie(engaged_counts, labels=['Not Engaged', 'Engaged'], autopct='%1.1f%%', 
        colors=colors, textprops={'fontsize': 12})
plt.title(f'Engagement Distribution (N={len(df)} Donors)')
# Add engagement rate annotation
engagement_rate = summary['Engagement_Rate'].mean() * 100  # Average across groups
plt.text(0, -1.2, f'Engagement Rate: {engagement_rate:.1f}%', 
         ha='center', fontsize=12, color='#333333')
plt.savefig('engagement_pie.png')
plt.close()

# Scatter plot: Interactions vs. Donation Amount (for engagement metrics tracking)
plt.figure(figsize=(8, 6))  # Set figure size for readability
colors = {0: '#FF4C4C', 1: '#4C78FF'}  # Red for Not Engaged, Blue for Engaged
for engaged_val in df['Engaged'].unique():
    subset = df[df['Engaged'] == engaged_val]
    plt.scatter(subset['NumInteractions'], subset['DonationAmount'], 
                c=colors[engaged_val], label='Engaged' if engaged_val == 1 else 'Not Engaged', 
                s=50, alpha=0.7)
plt.title('Donor Interactions vs. Donation Amount by Engagement Status')
plt.xlabel('Number of Interactions (e.g., Calls, Emails)')
plt.ylabel('Donation Amount ($)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('interactions_scatter.png')
plt.close()

# ML-Based Donor Segmentation: Use KMeans clustering on key features
# This optimizes donor segmentation for personalized communications
features = df[['Age', 'DonationAmount', 'NumInteractions', 'ResponseRate']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # 3 segments
df['Segment'] = kmeans.fit_predict(features_scaled)

# Export clustered data for further use (e.g., in CRM for targeted strategies)
df.to_csv('donor_clustered.csv', index=False)

print("Analysis complete. Check generated files for summaries and visualizations.")