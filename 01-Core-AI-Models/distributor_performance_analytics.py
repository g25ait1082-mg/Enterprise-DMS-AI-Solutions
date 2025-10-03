
"""
Distributor Performance Analytics & Scoring System
Author: Senior AI Manager - Unilever Digital Transformation
Purpose: Advanced analytics for distributor performance evaluation and optimization
Features: Performance scoring, churn prediction, growth opportunity identification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DistributorPerformanceAnalytics:
    def __init__(self):
        """
        Initialize Distributor Performance Analytics System
        Designed for enterprise DMS implementations
        """
        self.performance_model = None
        self.churn_model = None
        self.growth_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}

    def calculate_performance_metrics(self, sales_df, distributors_df, time_period_days=90):
        """
        Calculate comprehensive performance metrics for each distributor

        Args:
            sales_df: Sales transaction data
            distributors_df: Distributor master data
            time_period_days: Analysis period in days
        """
        print(f"📊 Calculating performance metrics for {len(distributors_df)} distributors")
        print(f"📅 Analysis period: {time_period_days} days")

        # Define analysis period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)

        # Filter sales data for the period
        period_sales = sales_df[
            pd.to_datetime(sales_df['order_date']) >= start_date
        ].copy()

        performance_metrics = []

        for _, distributor in distributors_df.iterrows():
            dist_id = distributor['distributor_id']
            dist_sales = period_sales[period_sales['distributor_id'] == dist_id]

            # Calculate key performance metrics
            metrics = {
                'distributor_id': dist_id,
                'region': distributor['region'],
                'category': distributor['category'],

                # Revenue Metrics
                'total_revenue': dist_sales['total_amount'].sum(),
                'avg_order_value': dist_sales['total_amount'].mean() if len(dist_sales) > 0 else 0,
                'order_count': len(dist_sales),
                'revenue_per_order': dist_sales['total_amount'].sum() / max(len(dist_sales), 1),

                # Volume Metrics
                'total_quantity': dist_sales['quantity'].sum(),
                'avg_quantity_per_order': dist_sales['quantity'].mean() if len(dist_sales) > 0 else 0,

                # Frequency Metrics
                'order_frequency': len(dist_sales) / (time_period_days / 7),  # Orders per week
                'days_since_last_order': (datetime.now() - pd.to_datetime(distributor['last_order_date'])).days,

                # Financial Health
                'credit_utilization': distributor['outstanding'] / distributor['credit_limit'],
                'outstanding_amount': distributor['outstanding'],

                # Product Diversity
                'unique_products': dist_sales['product_code'].nunique(),
                'product_diversity_score': dist_sales['product_code'].nunique() / max(dist_sales['product_code'].nunique().max(), 1) if len(period_sales) > 0 else 0,

                # Consistency Metrics
                'order_consistency': self.calculate_order_consistency(dist_sales),
                'revenue_growth_trend': self.calculate_growth_trend(dist_sales),

                # Operational Metrics
                'delivery_performance': self.calculate_delivery_performance(dist_sales),
                'cancellation_rate': (dist_sales['status'] == 'Cancelled').mean() if len(dist_sales) > 0 else 0,
            }

            performance_metrics.append(metrics)

        self.performance_df = pd.DataFrame(performance_metrics)

        # Calculate performance scores
        self.performance_df['performance_score'] = self.calculate_composite_score()

        print(f"✅ Performance metrics calculated for {len(self.performance_df)} distributors")
        return self.performance_df

    def calculate_order_consistency(self, dist_sales):
        """Calculate order consistency score (0-1)"""
        if len(dist_sales) < 2:
            return 0

        # Calculate coefficient of variation for order amounts
        cv_amount = dist_sales['total_amount'].std() / dist_sales['total_amount'].mean() if dist_sales['total_amount'].mean() > 0 else 1

        # Consistency score (lower CV = higher consistency)
        consistency = max(0, 1 - min(cv_amount, 1))
        return consistency

    def calculate_growth_trend(self, dist_sales):
        """Calculate revenue growth trend"""
        if len(dist_sales) < 4:
            return 0

        # Sort by date and calculate week-over-week growth
        dist_sales_sorted = dist_sales.sort_values('order_date')
        weekly_revenue = dist_sales_sorted.groupby(pd.Grouper(key='order_date', freq='W'))['total_amount'].sum()

        if len(weekly_revenue) < 2:
            return 0

        # Calculate trend (simple linear regression slope)
        x = np.arange(len(weekly_revenue))
        y = weekly_revenue.values

        if len(x) > 1 and np.std(x) > 0:
            correlation = np.corrcoef(x, y)[0, 1]
            return correlation if not np.isnan(correlation) else 0
        return 0

    def calculate_delivery_performance(self, dist_sales):
        """Calculate delivery performance score"""
        if len(dist_sales) == 0:
            return 0

        # Percentage of delivered orders
        delivered_rate = (dist_sales['status'] == 'Delivered').mean()
        return delivered_rate

    def calculate_composite_score(self):
        """Calculate composite performance score (0-100)"""
        # Normalize key metrics to 0-1 scale
        revenue_score = self.min_max_normalize(self.performance_df['total_revenue'])
        frequency_score = self.min_max_normalize(self.performance_df['order_frequency'])
        consistency_score = self.performance_df['order_consistency']
        growth_score = (self.performance_df['revenue_growth_trend'] + 1) / 2  # Convert -1,1 to 0,1
        delivery_score = self.performance_df['delivery_performance']
        diversity_score = self.performance_df['product_diversity_score']

        # Credit utilization penalty (lower is better)
        credit_penalty = 1 - np.clip(self.performance_df['credit_utilization'], 0, 1)

        # Weighted composite score
        composite_score = (
            revenue_score * 0.25 +
            frequency_score * 0.20 +
            consistency_score * 0.15 +
            growth_score * 0.15 +
            delivery_score * 0.10 +
            diversity_score * 0.10 +
            credit_penalty * 0.05
        ) * 100

        return np.round(composite_score, 2)

    def min_max_normalize(self, series):
        """Min-max normalization to 0-1 scale"""
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)

    def segment_distributors(self, n_clusters=5):
        """
        Segment distributors using K-means clustering
        Creates actionable business segments
        """
        print(f"🎯 Segmenting distributors into {n_clusters} clusters")

        # Select features for clustering
        clustering_features = [
            'total_revenue', 'order_frequency', 'avg_order_value',
            'product_diversity_score', 'order_consistency', 'credit_utilization'
        ]

        # Prepare data
        X = self.performance_df[clustering_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.performance_df['segment'] = kmeans.fit_predict(X_scaled)

        # Create segment profiles
        segment_profiles = self.performance_df.groupby('segment').agg({
            'total_revenue': ['mean', 'count'],
            'performance_score': 'mean',
            'order_frequency': 'mean',
            'credit_utilization': 'mean'
        }).round(2)

        # Assign business-friendly segment names
        segment_names = {
            0: 'Star Performers',
            1: 'Growth Potential',
            2: 'Steady Contributors',
            3: 'At Risk',
            4: 'New/Developing'
        }

        self.performance_df['segment_name'] = self.performance_df['segment'].map(
            lambda x: segment_names.get(x, f'Segment_{x}')
        )

        print("✅ Distributor segmentation completed")
        print("\n📊 Segment Distribution:")
        segment_distribution = self.performance_df['segment_name'].value_counts()
        for segment, count in segment_distribution.items():
            print(f"   {segment}: {count} distributors")

        return segment_profiles

    def predict_churn_risk(self, target_col='churn_risk'):
        """
        Train churn prediction model
        Identifies distributors at risk of churning
        """
        print("🔮 Training churn prediction model")

        # Create churn risk target (simplified logic)
        # In production, use historical churn data
        self.performance_df['churn_risk'] = (
            (self.performance_df['days_since_last_order'] > 30) |
            (self.performance_df['performance_score'] < 30) |
            (self.performance_df['credit_utilization'] > 0.8)
        ).astype(int)

        # Select features for churn prediction
        churn_features = [
            'total_revenue', 'order_frequency', 'days_since_last_order',
            'credit_utilization', 'performance_score', 'order_consistency',
            'cancellation_rate', 'delivery_performance'
        ]

        X = self.performance_df[churn_features].fillna(0)
        y = self.performance_df['churn_risk']

        if y.sum() > 0 and len(X) > 10:  # Ensure we have positive cases
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train Random Forest model
            self.churn_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )

            self.churn_model.fit(X_train, y_train)

            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': churn_features,
                'importance': self.churn_model.feature_importances_
            }).sort_values('importance', ascending=False)

            self.feature_importance['churn'] = feature_importance

            # Predict churn probabilities
            churn_probabilities = self.churn_model.predict_proba(X)[:, 1]
            self.performance_df['churn_probability'] = churn_probabilities

            # Evaluate model
            y_pred = self.churn_model.predict(X_test)
            print("\n📈 Churn Prediction Model Performance:")
            print(classification_report(y_test, y_pred))

            print(f"✅ Churn model trained - {y.sum()} at-risk distributors identified")
        else:
            print("⚠️  Insufficient data for churn prediction")

    def identify_growth_opportunities(self):
        """
        Identify growth opportunities for each distributor
        Provides actionable recommendations
        """
        print("🚀 Identifying growth opportunities")

        opportunities = []

        for _, dist in self.performance_df.iterrows():
            dist_opportunities = []

            # Low frequency opportunity
            if dist['order_frequency'] < 2:  # Less than 2 orders per week
                dist_opportunities.append({
                    'type': 'Increase Order Frequency',
                    'current_value': dist['order_frequency'],
                    'target_value': 3.0,
                    'potential_impact': 'High',
                    'recommendation': 'Implement automated reorder system'
                })

            # Low product diversity
            if dist['unique_products'] < 3:
                dist_opportunities.append({
                    'type': 'Expand Product Range',
                    'current_value': dist['unique_products'],
                    'target_value': 5.0,
                    'potential_impact': 'Medium',
                    'recommendation': 'Cross-sell complementary products'
                })

            # High credit utilization
            if dist['credit_utilization'] > 0.7:
                dist_opportunities.append({
                    'type': 'Optimize Credit Management',
                    'current_value': dist['credit_utilization'],
                    'target_value': 0.5,
                    'potential_impact': 'High',
                    'recommendation': 'Implement payment terms optimization'
                })

            # Low order value
            avg_benchmark = self.performance_df['avg_order_value'].median()
            if dist['avg_order_value'] < avg_benchmark * 0.8:
                dist_opportunities.append({
                    'type': 'Increase Order Value',
                    'current_value': dist['avg_order_value'],
                    'target_value': avg_benchmark,
                    'potential_impact': 'Medium',
                    'recommendation': 'Volume-based incentives and bundling'
                })

            opportunities.append({
                'distributor_id': dist['distributor_id'],
                'segment': dist.get('segment_name', 'Unknown'),
                'performance_score': dist['performance_score'],
                'opportunities': dist_opportunities,
                'priority_score': len(dist_opportunities) * (100 - dist['performance_score']) / 100
            })

        self.growth_opportunities = pd.DataFrame(opportunities)

        print(f"✅ Growth opportunities identified for {len(self.growth_opportunities)} distributors")

        # Top opportunities
        top_opportunities = self.growth_opportunities.nlargest(5, 'priority_score')
        print("\n🎯 Top Growth Opportunities:")
        for _, opp in top_opportunities.iterrows():
            print(f"   {opp['distributor_id']}: {len(opp['opportunities'])} opportunities (Priority: {opp['priority_score']:.2f})")

        return self.growth_opportunities

    def generate_distributor_report(self, distributor_id):
        """Generate comprehensive report for a specific distributor"""
        dist_data = self.performance_df[self.performance_df['distributor_id'] == distributor_id]

        if len(dist_data) == 0:
            return "Distributor not found"

        dist = dist_data.iloc[0]

        report = f"""

🏢 DISTRIBUTOR PERFORMANCE REPORT
{'='*50}

📋 Basic Information:
   • Distributor ID: {dist['distributor_id']}
   • Region: {dist['region']}
   • Category: {dist['category']}
   • Segment: {dist.get('segment_name', 'Unknown')}

📊 Performance Metrics:
   • Overall Score: {dist['performance_score']:.1f}/100
   • Total Revenue: ₹{dist['total_revenue']:,.0f}
   • Order Frequency: {dist['order_frequency']:.1f} orders/week
   • Average Order Value: ₹{dist['avg_order_value']:,.0f}
   • Product Diversity: {dist['unique_products']} unique products

🔍 Risk Assessment:
   • Churn Risk: {'HIGH' if dist.get('churn_probability', 0) > 0.5 else 'LOW'}
   • Credit Utilization: {dist['credit_utilization']:.1%}
   • Days Since Last Order: {dist['days_since_last_order']} days

💡 Growth Opportunities:
        """

        # Add opportunities if available
        if hasattr(self, 'growth_opportunities'):
            dist_opportunities = self.growth_opportunities[
                self.growth_opportunities['distributor_id'] == distributor_id
            ]

            if len(dist_opportunities) > 0:
                opportunities = dist_opportunities.iloc[0]['opportunities']
                for i, opp in enumerate(opportunities, 1):
                    report += f"\n   {i}. {opp['type']}: {opp['recommendation']}"

        report += "\n\n✅ Report generated successfully\n"

        return report

def demonstrate_distributor_analytics():
    """Demonstrate distributor analytics with sample data"""
    print("🚀 Distributor Performance Analytics - Enterprise DMS Solution")
    print("="*70)

    # Load sample data (assuming CSV files exist)
    try:
        distributors_df = pd.read_csv('distributors_data.csv')
        sales_df = pd.read_csv('sales_data.csv')

        print(f"📊 Loaded {len(distributors_df)} distributors and {len(sales_df)} sales records")

        # Initialize analytics system
        analytics = DistributorPerformanceAnalytics()

        # Calculate performance metrics
        performance_data = analytics.calculate_performance_metrics(sales_df, distributors_df)

        # Segment distributors
        segment_profiles = analytics.segment_distributors()

        # Predict churn risk
        analytics.predict_churn_risk()

        # Identify growth opportunities
        growth_opportunities = analytics.identify_growth_opportunities()

        # Generate sample report
        sample_report = analytics.generate_distributor_report('DIST_001')
        print(sample_report)

        # Summary statistics
        print("\n📈 ANALYTICS SUMMARY")
        print("-" * 30)
        print(f"Average Performance Score: {performance_data['performance_score'].mean():.1f}")
        print(f"High Performers (>80 score): {(performance_data['performance_score'] > 80).sum()}")
        print(f"At Risk Distributors: {(performance_data.get('churn_probability', pd.Series([0])) > 0.5).sum()}")
        print(f"Growth Opportunities Identified: {sum([len(opp['opportunities']) for _, opp in growth_opportunities.iterrows()])}")

        return analytics

    except FileNotFoundError:
        print("⚠️  Sample data files not found. Please run the data generation script first.")
        return None

if __name__ == "__main__":
    # Demonstrate analytics system
    analytics_system = demonstrate_distributor_analytics()

    if analytics_system:
        print("\n✅ Distributor Analytics System Ready")
        print("🔗 Integration Points:")
        print("   • SAP ERP for real-time data sync")
        print("   • Power BI for executive dashboards") 
        print("   • Salesforce for CRM integration")
        print("   • Mobile SFA app for field force insights")

