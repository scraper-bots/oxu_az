import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure matplotlib for better text rendering
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

print("Loading data...")
df = pd.read_csv('oxu_articles.csv')

# Clean data
df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0).astype(int)
df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0).astype(int)
df['dislikes'] = pd.to_numeric(df['dislikes'], errors='coerce').fillna(0).astype(int)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Add calculated metrics
df['total_engagement'] = df['likes'] + df['dislikes']
df['engagement_rate'] = df['total_engagement'] / (df['views'] + 1) * 100  # +1 to avoid division by zero
df['like_ratio'] = df['likes'] / (df['total_engagement'] + 1) * 100  # Percentage of likes
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()

print(f"Total articles analyzed: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print("\n" + "="*80)

# ============================================================================
# CHART 1: Average Views by Category - What content gets most traffic?
# ============================================================================
print("\nGenerating Chart 1: Views by Category...")

fig, ax = plt.subplots(figsize=(14, 8))

category_stats = df.groupby('category').agg({
    'views': ['mean', 'sum', 'count']
}).round(0)

category_stats.columns = ['avg_views', 'total_views', 'article_count']
category_stats = category_stats.sort_values('avg_views', ascending=False)

bars = ax.barh(category_stats.index, category_stats['avg_views'], color=sns.color_palette("rocket", len(category_stats)))

# Add value labels on bars
for i, (idx, row) in enumerate(category_stats.iterrows()):
    ax.text(row['avg_views'] + 20, i, f"{int(row['avg_views'])} views\n({int(row['article_count'])} articles)",
            va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Average Views per Article', fontsize=13, fontweight='bold')
ax.set_title('📊 Content Performance: Average Views by Category\n\n'
             'Which categories drive the most traffic?',
             fontsize=16, fontweight='bold', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('chart1_views_by_category.png', dpi=300, bbox_inches='tight')
print("✓ Saved: chart1_views_by_category.png")
plt.close()

# ============================================================================
# CHART 2: Engagement Quality by Category - Which content gets best engagement?
# ============================================================================
print("Generating Chart 2: Engagement Quality by Category...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: Engagement Rate
engagement_by_cat = df.groupby('category').agg({
    'engagement_rate': 'mean',
    'like_ratio': 'mean'
}).sort_values('engagement_rate', ascending=False)

bars1 = ax1.barh(engagement_by_cat.index, engagement_by_cat['engagement_rate'],
                 color=sns.color_palette("mako", len(engagement_by_cat)))

for i, (idx, row) in enumerate(engagement_by_cat.iterrows()):
    ax1.text(row['engagement_rate'] + 0.05, i, f"{row['engagement_rate']:.2f}%",
             va='center', fontsize=10, fontweight='bold')

ax1.set_xlabel('Engagement Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Engagement Rate by Category\n(Higher = More Interactive)', fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: Like Ratio (Quality of Engagement)
engagement_by_cat_sorted = engagement_by_cat.sort_values('like_ratio', ascending=False)

bars2 = ax2.barh(engagement_by_cat_sorted.index, engagement_by_cat_sorted['like_ratio'],
                 color=sns.color_palette("viridis", len(engagement_by_cat_sorted)))

for i, (idx, row) in enumerate(engagement_by_cat_sorted.iterrows()):
    ax2.text(row['like_ratio'] + 1, i, f"{row['like_ratio']:.1f}%",
             va='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('Positive Reaction Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Content Approval Rate by Category\n(Higher = More Liked)', fontsize=13, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('💬 Engagement Quality: How Readers React to Content',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('chart2_engagement_quality.png', dpi=300, bbox_inches='tight')
print("✓ Saved: chart2_engagement_quality.png")
plt.close()

# ============================================================================
# CHART 3: Publishing Time Analysis - When to publish for maximum views?
# ============================================================================
print("Generating Chart 3: Best Publishing Times...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# By Hour of Day
hourly_stats = df.groupby('hour').agg({
    'views': 'mean',
    'engagement_rate': 'mean'
}).reset_index()

ax1.plot(hourly_stats['hour'], hourly_stats['views'], marker='o', linewidth=2.5,
         markersize=8, color='#e74c3c')
ax1.fill_between(hourly_stats['hour'], hourly_stats['views'], alpha=0.3, color='#e74c3c')
ax1.set_xlabel('Hour of Day (24-hour format)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Views', fontsize=12, fontweight='bold')
ax1.set_title('Best Publishing Hours\n(Peak Traffic Times)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, 24))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Mark peak hours
peak_hour = hourly_stats.loc[hourly_stats['views'].idxmax(), 'hour']
peak_views = hourly_stats['views'].max()
ax1.axvline(x=peak_hour, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(peak_hour, peak_views * 1.05, f'Peak: {int(peak_hour)}:00\n({int(peak_views)} views)',
         ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# By Day of Week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_stats = df.groupby('day_of_week')['views'].mean().reindex(day_order)

bars = ax2.bar(range(len(daily_stats)), daily_stats.values,
               color=sns.color_palette("coolwarm", len(daily_stats)))

for i, (day, views) in enumerate(daily_stats.items()):
    if pd.notna(views):
        ax2.text(i, views + 10, f'{int(views)}', ha='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Views', fontsize=12, fontweight='bold')
ax2.set_title('Best Publishing Days\n(Weekly Pattern)', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(day_order)))
ax2.set_xticklabels([d[:3] for d in day_order])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('⏰ Timing Strategy: When to Publish for Maximum Impact',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('chart3_publishing_times.png', dpi=300, bbox_inches='tight')
print("✓ Saved: chart3_publishing_times.png")
plt.close()

# ============================================================================
# CHART 4: Tag Performance - Which content types drive more views?
# ============================================================================
print("Generating Chart 4: Content Tag Analysis...")

# Parse tags
df['tags_list'] = df['tags'].fillna('').apply(lambda x: [t.strip() for t in x.split(',') if t.strip()])

# Explode tags to get one row per tag
tag_data = []
for _, row in df.iterrows():
    for tag in row['tags_list']:
        if tag:
            tag_data.append({
                'tag': tag,
                'views': row['views'],
                'engagement_rate': row['engagement_rate'],
                'likes': row['likes']
            })

if tag_data:
    tag_df = pd.DataFrame(tag_data)
    tag_stats = tag_df.groupby('tag').agg({
        'views': ['mean', 'count'],
        'engagement_rate': 'mean'
    })

    tag_stats.columns = ['avg_views', 'count', 'engagement_rate']
    tag_stats = tag_stats[tag_stats['count'] >= 10]  # Only tags with 10+ articles
    tag_stats = tag_stats.sort_values('avg_views', ascending=False).head(10)

    if len(tag_stats) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))

        bars = ax.barh(range(len(tag_stats)), tag_stats['avg_views'],
                       color=sns.color_palette("Set2", len(tag_stats)))

        ax.set_yticks(range(len(tag_stats)))
        ax.set_yticklabels(tag_stats.index)

        for i, (idx, row) in enumerate(tag_stats.iterrows()):
            ax.text(row['avg_views'] + 20, i,
                   f"{int(row['avg_views'])} views\n({int(row['count'])} articles)",
                   va='center', fontsize=10, fontweight='bold')

        ax.set_xlabel('Average Views per Article', fontsize=13, fontweight='bold')
        ax.set_title('🏷️ Content Type Performance: Views by Tag\n\n'
                     'Which content formats drive the most traffic?',
                     fontsize=16, fontweight='bold', pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig('chart4_tag_performance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: chart4_tag_performance.png")
        plt.close()
    else:
        print("⚠ Not enough tag data for Chart 4")
else:
    print("⚠ No tag data available for Chart 4")

# ============================================================================
# CHART 5: Top Performing Articles - Learn from the best
# ============================================================================
print("Generating Chart 5: Top Performing Articles...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Top 15 by Views
top_by_views = df.nlargest(15, 'views')[['title', 'views', 'category', 'engagement_rate']]

bars1 = ax1.barh(range(len(top_by_views)), top_by_views['views'],
                 color=sns.color_palette("rocket_r", len(top_by_views)))

# Truncate long titles
titles_short = [title[:70] + '...' if len(title) > 70 else title for title in top_by_views['title']]
ax1.set_yticks(range(len(top_by_views)))
ax1.set_yticklabels(titles_short, fontsize=9)

for i, (idx, row) in enumerate(top_by_views.iterrows()):
    ax1.text(row['views'] + 50, i, f"{int(row['views'])} views",
             va='center', fontsize=9, fontweight='bold')

ax1.set_xlabel('Total Views', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 Most Viewed Articles', fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.invert_yaxis()

# Top 15 by Engagement Rate (with min 50 views filter)
top_by_engagement = df[df['views'] >= 50].nlargest(15, 'engagement_rate')[['title', 'engagement_rate', 'views', 'category']]

bars2 = ax2.barh(range(len(top_by_engagement)), top_by_engagement['engagement_rate'],
                 color=sns.color_palette("viridis_r", len(top_by_engagement)))

titles_short2 = [title[:70] + '...' if len(title) > 70 else title for title in top_by_engagement['title']]
ax2.set_yticks(range(len(top_by_engagement)))
ax2.set_yticklabels(titles_short2, fontsize=9)

for i, (idx, row) in enumerate(top_by_engagement.iterrows()):
    ax2.text(row['engagement_rate'] + 0.2, i, f"{row['engagement_rate']:.1f}%",
             va='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('Engagement Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Top 15 Most Engaging Articles (min. 50 views)', fontsize=13, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.invert_yaxis()

fig.suptitle('🏆 Success Stories: Learn from Top Performing Content',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('chart5_top_articles.png', dpi=300, bbox_inches='tight')
print("✓ Saved: chart5_top_articles.png")
plt.close()

# ============================================================================
# CHART 6: Category Distribution & Content Strategy
# ============================================================================
print("Generating Chart 6: Content Mix Analysis...")

fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.3])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Pie chart: Article distribution
category_counts = df['category'].value_counts()
colors = sns.color_palette("husl", len(category_counts))

# Create pie chart with legend to avoid label overlap
wedges, texts, autotexts = ax1.pie(
    category_counts.values,
    labels=None,  # Don't show labels on pie - use legend instead
    autopct='%1.1f%%',
    startangle=45,
    colors=colors,
    pctdistance=0.85,
    textprops={'fontsize': 10}
)

# Make percentage text bold and white for readability
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

# Add clean legend with article counts
legend_labels = [f'{cat} ({int(count)} articles)'
                for cat, count in category_counts.items()]
ax1.legend(wedges, legend_labels,
          title="Category",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=10,
          title_fontsize=11,
          frameon=True)

ax1.set_title('Content Distribution\n(By Number of Articles)', fontsize=13, fontweight='bold', pad=15)

# Scatter: Views vs Engagement
for category in df['category'].unique():
    cat_data = df[df['category'] == category]
    ax2.scatter(cat_data['views'], cat_data['engagement_rate'],
               label=category, alpha=0.6, s=50)

ax2.set_xlabel('Views', fontsize=12, fontweight='bold')
ax2.set_ylabel('Engagement Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Views vs Engagement by Category\n(Bubble = Article)', fontsize=13, fontweight='bold')
ax2.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('📈 Content Strategy Overview: Distribution & Performance',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('chart6_content_strategy.png', dpi=300, bbox_inches='tight')
print("✓ Saved: chart6_content_strategy.png")
plt.close()

# ============================================================================
# Generate Summary Report
# ============================================================================
print("\n" + "="*80)
print("📊 ANALYSIS SUMMARY")
print("="*80)

print(f"\n1. OVERALL STATISTICS:")
print(f"   • Total Articles: {len(df):,}")
print(f"   • Total Views: {df['views'].sum():,}")
print(f"   • Average Views per Article: {df['views'].mean():.0f}")
print(f"   • Average Engagement Rate: {df['engagement_rate'].mean():.2f}%")

print(f"\n2. BEST PERFORMING CATEGORY (by avg views):")
best_cat = category_stats.index[0]
best_cat_views = category_stats.iloc[0]['avg_views']
print(f"   • {best_cat}: {best_cat_views:.0f} avg views per article")

print(f"\n3. MOST ENGAGING CATEGORY:")
best_eng_cat = engagement_by_cat.index[0]
best_eng_rate = engagement_by_cat.iloc[0]['engagement_rate']
print(f"   • {best_eng_cat}: {best_eng_rate:.2f}% engagement rate")

if not hourly_stats.empty:
    print(f"\n4. BEST PUBLISHING TIME:")
    print(f"   • Hour: {int(peak_hour)}:00 ({int(peak_views)} avg views)")

if not daily_stats.empty and not daily_stats.isna().all():
    best_day = daily_stats.idxmax()
    best_day_views = daily_stats.max()
    print(f"   • Day: {best_day} ({int(best_day_views)} avg views)")

print(f"\n5. TOP ARTICLE:")
top_article = df.nlargest(1, 'views').iloc[0]
print(f"   • Title: {top_article['title'][:100]}...")
print(f"   • Views: {int(top_article['views']):,}")
print(f"   • Category: {top_article['category']}")

print("\n" + "="*80)
print("✅ Analysis complete! All charts saved successfully.")
print("="*80)
