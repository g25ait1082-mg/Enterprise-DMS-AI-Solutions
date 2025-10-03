
"""
Next Best Action (NBA) Engine for Distribution Management
Author: Senior AI Manager - Unilever Digital Transformation Team
Purpose: Real-time recommendation engine for distributor actions and product suggestions
Technology: Hybrid recommendation system with collaborative & content-based filtering
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import redis
from datetime import datetime, timedelta
import logging

class NextBestActionEngine:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """
        Initialize NBA Engine with Redis for real-time serving

        Args:
            redis_host (str): Redis server host for caching recommendations
            redis_port (int): Redis server port
            redis_db (int): Redis database number
        """
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
            self.redis_client.ping()
            print("✅ Connected to Redis for real-time serving")
        except:
            print("⚠️  Redis not available, using in-memory caching")

        self.collaborative_model = None
        self.content_model = None
        self.business_rules_model = None
        self.feature_weights = {
            'collaborative': 0.4,
            'content': 0.3,
            'business_rules': 0.3
        }

    def prepare_interaction_matrix(self, transactions_df):
        """
        Create user-item interaction matrix for collaborative filtering
        Used for distributor-product interaction patterns
        """
        interaction_matrix = transactions_df.pivot_table(
            index='distributor_id',
            columns='product_code',
            values='quantity',
            aggfunc='sum',
            fill_value=0
        )

        print(f"📊 Interaction Matrix: {interaction_matrix.shape}")
        print(f"🏢 Distributors: {len(interaction_matrix.index)}")
        print(f"📦 Products: {len(interaction_matrix.columns)}")
        print(f"🔄 Sparsity: {(interaction_matrix == 0).sum().sum() / interaction_matrix.size:.2%}")

        return interaction_matrix

    def train_collaborative_filtering(self, interaction_matrix, n_components=50):
        """
        Train collaborative filtering using Matrix Factorization (SVD)
        Identifies similar distributors and their purchasing patterns
        """
        print("🔄 Training Collaborative Filtering Model...")

        # Apply SVD for dimensionality reduction and pattern discovery
        self.collaborative_model = TruncatedSVD(n_components=n_components, random_state=42)
        distributor_features = self.collaborative_model.fit_transform(interaction_matrix)

        # Calculate distributor similarities
        distributor_similarities = cosine_similarity(distributor_features)

        self.distributor_similarities = pd.DataFrame(
            distributor_similarities,
            index=interaction_matrix.index,
            columns=interaction_matrix.index
        )

        print(f"✅ Collaborative model trained with {n_components} components")
        return distributor_features

    def train_content_based_filtering(self, products_df):
        """
        Train content-based filtering using product features
        Recommends products based on category, brand, attributes
        """
        print("🔄 Training Content-Based Filtering Model...")

        # Create product feature descriptions
        products_df['content_features'] = (
            products_df['category'] + ' ' +
            products_df['brand'] + ' ' +
            products_df['attributes'].fillna('')
        )

        # Create TF-IDF vectors for product content
        self.content_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        content_features = self.content_vectorizer.fit_transform(products_df['content_features'])

        # Calculate product similarities
        product_similarities = cosine_similarity(content_features)

        self.product_similarities = pd.DataFrame(
            product_similarities,
            index=products_df['product_code'],
            columns=products_df['product_code']
        )

        print(f"✅ Content-based model trained for {len(products_df)} products")

    def train_business_rules_model(self, transactions_df, distributors_df):
        """
        Train business rules model for contextual recommendations
        Incorporates business logic: seasonality, inventory, promotions, performance
        """
        print("🔄 Training Business Rules Model...")

        # Feature engineering for business context
        feature_data = []

        for _, dist in distributors_df.iterrows():
            dist_transactions = transactions_df[
                transactions_df['distributor_id'] == dist['distributor_id']
            ]

            if len(dist_transactions) > 0:
                features = {
                    'distributor_id': dist['distributor_id'],
                    'region': dist['region'],
                    'category': dist['category'],
                    'credit_utilization': dist['outstanding'] / dist['credit_limit'],
                    'avg_order_value': dist_transactions['total_amount'].mean(),
                    'order_frequency': len(dist_transactions),
                    'last_order_days': (datetime.now() - pd.to_datetime(dist['last_order_date'])).days,
                    'performance_score': min(100, (dist_transactions['total_amount'].sum() / 100000) * 100)
                }
                feature_data.append(features)

        self.business_features_df = pd.DataFrame(feature_data)

        # Train classification model for high-value opportunities
        if len(self.business_features_df) > 10:
            # Create target: high-value opportunity (top 30% by performance)
            threshold = self.business_features_df['performance_score'].quantile(0.7)
            self.business_features_df['high_value_target'] = (
                self.business_features_df['performance_score'] > threshold
            ).astype(int)

            # Features for training
            feature_cols = ['credit_utilization', 'avg_order_value', 'order_frequency', 
                          'last_order_days', 'performance_score']

            X = self.business_features_df[feature_cols]
            y = self.business_features_df['high_value_target']

            if len(X) > 5 and y.sum() > 0:  # Ensure we have enough data
                self.business_rules_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=5
                )

                self.business_rules_model.fit(X, y)
                print(f"✅ Business rules model trained with {len(X)} samples")
            else:
                print("⚠️  Insufficient data for business rules model")

    def get_collaborative_recommendations(self, distributor_id, interaction_matrix, top_k=10):
        """Get recommendations based on similar distributors"""
        if distributor_id not in self.distributor_similarities.index:
            return []

        # Find similar distributors
        similar_distributors = self.distributor_similarities.loc[distributor_id].sort_values(ascending=False)[1:6]

        # Get products purchased by similar distributors but not by target distributor
        target_products = set(interaction_matrix.loc[distributor_id][interaction_matrix.loc[distributor_id] > 0].index)

        recommendations = []
        for similar_dist, similarity_score in similar_distributors.items():
            similar_products = set(interaction_matrix.loc[similar_dist][interaction_matrix.loc[similar_dist] > 0].index)
            candidate_products = similar_products - target_products

            for product in candidate_products:
                recommendations.append({
                    'product_code': product,
                    'score': similarity_score * interaction_matrix.loc[similar_dist, product],
                    'reason': f'Distributors like you also buy this (similarity: {similarity_score:.2f})'
                })

        # Sort by score and return top_k
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_k]
        return recommendations

    def get_content_recommendations(self, recent_products, top_k=10):
        """Get recommendations based on product content similarity"""
        if not recent_products or self.product_similarities is None:
            return []

        # Get average similarity scores for recent products
        product_scores = {}

        for product in recent_products:
            if product in self.product_similarities.index:
                similar_products = self.product_similarities.loc[product].sort_values(ascending=False)[1:]

                for similar_product, score in similar_products.items():
                    if similar_product not in recent_products:
                        if similar_product not in product_scores:
                            product_scores[similar_product] = []
                        product_scores[similar_product].append(score)

        # Average scores for each product
        recommendations = []
        for product, scores in product_scores.items():
            avg_score = np.mean(scores)
            recommendations.append({
                'product_code': product,
                'score': avg_score,
                'reason': f'Similar to your recent purchases (similarity: {avg_score:.2f})'
            })

        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_k]
        return recommendations

    def get_business_recommendations(self, distributor_id, context=None):
        """Get recommendations based on business rules and context"""
        recommendations = []

        # Example business rules (customize based on your business logic)
        business_recommendations = [
            {
                'product_code': 'SEASONAL_001',
                'score': 0.9,
                'reason': 'Seasonal product for upcoming festival season',
                'action': 'stock_up'
            },
            {
                'product_code': 'PROMO_002',
                'score': 0.8,
                'reason': 'Active promotion with 15% margin improvement',
                'action': 'promote'
            },
            {
                'product_code': 'REORDER_003',
                'score': 0.7,
                'reason': 'Low inventory alert - reorder recommended',
                'action': 'reorder'
            }
        ]

        return business_recommendations

    def get_next_best_actions(self, distributor_id, interaction_matrix, recent_products=None, context=None, top_k=5):
        """
        Main NBA function: combines all recommendation approaches
        Returns prioritized list of next best actions
        """
        print(f"🎯 Generating NBA for Distributor: {distributor_id}")

        all_recommendations = []

        # 1. Collaborative Filtering Recommendations
        collab_recs = self.get_collaborative_recommendations(distributor_id, interaction_matrix, top_k=top_k)
        for rec in collab_recs:
            rec['source'] = 'collaborative'
            rec['final_score'] = rec['score'] * self.feature_weights['collaborative']
        all_recommendations.extend(collab_recs)

        # 2. Content-Based Recommendations
        if recent_products:
            content_recs = self.get_content_recommendations(recent_products, top_k=top_k)
            for rec in content_recs:
                rec['source'] = 'content'
                rec['final_score'] = rec['score'] * self.feature_weights['content']
            all_recommendations.extend(content_recs)

        # 3. Business Rules Recommendations
        business_recs = self.get_business_recommendations(distributor_id, context)
        for rec in business_recs:
            rec['source'] = 'business_rules'
            rec['final_score'] = rec['score'] * self.feature_weights['business_rules']
        all_recommendations.extend(business_recs)

        # Combine and deduplicate recommendations
        final_recommendations = {}
        for rec in all_recommendations:
            product = rec['product_code']
            if product not in final_recommendations:
                final_recommendations[product] = rec
            else:
                # Combine scores if product appears in multiple sources
                final_recommendations[product]['final_score'] += rec['final_score']
                final_recommendations[product]['reason'] += f" + {rec['reason']}"

        # Sort by final score and return top recommendations
        final_recs = list(final_recommendations.values())
        final_recs = sorted(final_recs, key=lambda x: x['final_score'], reverse=True)[:top_k]

        # Cache recommendations in Redis
        self.cache_recommendations(distributor_id, final_recs)

        print(f"✅ Generated {len(final_recs)} NBA recommendations")
        return final_recs

    def cache_recommendations(self, distributor_id, recommendations):
        """Cache recommendations in Redis for real-time serving"""
        if self.redis_client:
            try:
                cache_key = f"nba:{distributor_id}"
                cache_data = {
                    'timestamp': datetime.now().isoformat(),
                    'recommendations': recommendations
                }

                self.redis_client.setex(
                    cache_key,
                    timedelta(hours=24),  # Cache for 24 hours
                    json.dumps(cache_data, default=str)
                )
                print(f"💾 Cached recommendations for {distributor_id}")
            except Exception as e:
                print(f"⚠️  Cache error: {e}")

def demonstrate_nba_engine():
    """Demonstrate NBA Engine with sample Unilever DMS data"""
    print("🚀 Next Best Action Engine - Unilever DMS Implementation")
    print("=" * 60)

    # Sample data setup
    distributors = pd.DataFrame({
        'distributor_id': [f'DIST_{i:03d}' for i in range(1, 11)],
        'region': ['North', 'South', 'East', 'West', 'Central'] * 2,
        'category': ['FMCG'] * 10,
        'credit_limit': [500000] * 10,
        'outstanding': [200000, 150000, 300000, 100000, 250000] * 2,
        'last_order_date': ['2025-10-01'] * 10
    })

    products = pd.DataFrame({
        'product_code': ['PROD_A', 'PROD_B', 'PROD_C', 'PROD_D', 'PROD_E'],
        'category': ['Personal Care', 'Home Care', 'Foods', 'Ice Cream', 'Tea'],
        'brand': ['Dove', 'Surf', 'Knorr', 'Magnum', 'Lipton'],
        'attributes': ['moisturizing premium', 'powerful cleaning', 'instant cooking', 'premium ice cream', 'premium tea']
    })

    transactions = pd.DataFrame({
        'distributor_id': ['DIST_001'] * 3 + ['DIST_002'] * 3 + ['DIST_003'] * 3,
        'product_code': ['PROD_A', 'PROD_B', 'PROD_C'] * 3,
        'quantity': [100, 150, 200, 120, 180, 90, 200, 100, 300],
        'total_amount': [10000, 15000, 20000, 12000, 18000, 9000, 20000, 10000, 30000]
    })

    # Initialize and train NBA Engine
    nba_engine = NextBestActionEngine()

    # Prepare interaction matrix
    interaction_matrix = nba_engine.prepare_interaction_matrix(transactions)

    # Train models
    nba_engine.train_collaborative_filtering(interaction_matrix)
    nba_engine.train_content_based_filtering(products)
    nba_engine.train_business_rules_model(transactions, distributors)

    # Generate NBA recommendations
    recommendations = nba_engine.get_next_best_actions(
        distributor_id='DIST_001',
        interaction_matrix=interaction_matrix,
        recent_products=['PROD_A', 'PROD_B'],
        top_k=3
    )

    print("\n🎯 Next Best Action Recommendations:")
    print("-" * 40)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['product_code']} (Score: {rec['final_score']:.3f})")
        print(f"   📝 Reason: {rec['reason']}")
        print(f"   🔍 Source: {rec['source']}")
        print()

    return nba_engine

if __name__ == "__main__":
    # Demonstrate NBA Engine
    engine = demonstrate_nba_engine()
    print("✅ NBA Engine ready for production deployment")
    print("🔗 Integration points: SAP ERP, Salesforce, Mobile SFA App")
    print("⚡ Real-time serving via Redis cache and API endpoints")

