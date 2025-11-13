"""
E-Commerce AI Agent - Data Setup Script
This script loads Brazilian e-commerce CSV data into SQLite database
"""

import pandas as pd
import sqlite3
from pathlib import Path
import sys

def setup_database(data_folder="data", db_path="ecommerce.db"):
    """
    Load all CSV files into SQLite database
    
    Args:
        data_folder: Path to folder containing CSV files
        db_path: Path where SQLite database will be created
    """
    
    print("=" * 60)
    print("E-COMMERCE DATABASE SETUP")
    print("=" * 60)
    
    # Check if data folder exists
    data_path = Path(data_folder)
    if not data_path.exists():
        print(f"\n❌ Error: Data folder '{data_folder}' not found!")
        print("\nPlease download the dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/")
        print(f"Extract all CSV files to the '{data_folder}' folder\n")
        return False
    
    conn = sqlite3.connect(db_path)
    
    # Define files and table names
    files = {
        'olist_customers_dataset.csv': 'customers',
        'olist_geolocation_dataset.csv': 'geolocation',
        'olist_order_items_dataset.csv': 'order_items',
        'olist_order_payments_dataset.csv': 'order_payments',
        'olist_order_reviews_dataset.csv': 'order_reviews',
        'olist_orders_dataset.csv': 'orders',
        'olist_products_dataset.csv': 'products',
        'olist_sellers_dataset.csv': 'sellers',
        'product_category_name_translation.csv': 'category_translation'
    }
    
    print("\n📊 Loading datasets into SQLite database...\n")
    
    loaded_count = 0
    for file, table in files.items():
        file_path = data_path / file
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df.to_sql(table, conn, if_exists='replace', index=False)
                print(f"✓ Loaded {table:25s}: {len(df):,} rows, {len(df.columns)} columns")
                loaded_count += 1
            except Exception as e:
                print(f"✗ Error loading {file}: {str(e)}")
        else:
            print(f"⚠ Missing: {file}")
    
    conn.close()
    
    print(f"\n{'=' * 60}")
    print(f"✓ Database created successfully: {db_path}")
    print(f"✓ Loaded {loaded_count}/{len(files)} tables")
    print(f"{'=' * 60}\n")
    
    # Generate schema documentation
    if loaded_count > 0:
        generate_schema_docs(db_path)
        return True
    
    return False

def generate_schema_docs(db_path):
    """Create schema documentation for the AI agent"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_doc = """DATABASE SCHEMA FOR BRAZILIAN E-COMMERCE DATASET
================================================

This database contains data about 100,000 orders from 2016 to 2018 made at multiple marketplaces in Brazil.

"""
    
    # Table descriptions
    descriptions = {
        'customers': 'Customer information including location',
        'orders': 'Order details with timestamps and status',
        'order_items': 'Items within each order with pricing',
        'order_payments': 'Payment information for orders',
        'order_reviews': 'Customer reviews and ratings',
        'products': 'Product catalog with categories',
        'sellers': 'Seller information and location',
        'geolocation': 'Geographic coordinates for zip codes',
        'category_translation': 'Product category translations to English'
    }
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        schema_doc += f"\nTable: {table_name}\n"
        schema_doc += f"Description: {descriptions.get(table_name, 'No description')}\n"
        schema_doc += f"Row count: {row_count:,}\n"
        schema_doc += "Columns:\n"
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            schema_doc += f"  • {col_name} ({col_type})\n"
        schema_doc += "\n" + "-" * 60 + "\n"
    
    # Add common query examples
    schema_doc += """
COMMON QUERIES EXAMPLES:
========================

1. Get order statistics:
   SELECT COUNT(*) as total_orders, 
          AVG(payment_value) as avg_value 
   FROM order_payments;

2. Top product categories:
   SELECT p.product_category_name, 
          COUNT(*) as order_count 
   FROM order_items oi 
   JOIN products p ON oi.product_id = p.product_id 
   GROUP BY p.product_category_name 
   ORDER BY order_count DESC LIMIT 10;

3. Customer satisfaction by state:
   SELECT c.customer_state, 
          AVG(r.review_score) as avg_rating 
   FROM customers c 
   JOIN orders o ON c.customer_id = o.customer_id 
   JOIN order_reviews r ON o.order_id = r.order_id 
   GROUP BY c.customer_state 
   ORDER BY avg_rating DESC;

4. Monthly order trends:
   SELECT strftime('%Y-%m', order_purchase_timestamp) as month,
          COUNT(*) as orders 
   FROM orders 
   GROUP BY month 
   ORDER BY month;
"""
    
    conn.close()
    
    # Save to file
    with open("schema_docs.txt", "w") as f:
        f.write(schema_doc)
    
    print("✓ Schema documentation created: schema_docs.txt\n")

def verify_database(db_path="ecommerce.db"):
    """Verify database was created correctly"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table count
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
        table_count = cursor.fetchone()[0]
        
        print(f"✓ Database verification successful!")
        print(f"  - Found {table_count} tables")
        
        # Sample query
        cursor.execute("SELECT COUNT(*) FROM orders")
        order_count = cursor.fetchone()[0]
        print(f"  - Total orders: {order_count:,}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Database verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n🚀 Starting database setup...\n")
    
    success = setup_database()
    
    if success:
        verify_database()
        print("\n✅ Setup complete! You can now run: streamlit run app.py\n")
    else:
        print("\n❌ Setup failed. Please check the error messages above.\n")
        sys.exit(1)
