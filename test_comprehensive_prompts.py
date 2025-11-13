"""
Comprehensive test suite: 30 prompts from easy to advanced
Tests SQL queries, joins, aggregations, translations, and date handling
"""

from agent import query_agent, query_agent_with_followup
import time


test_prompts = [
    {
        "level": "Easy",
        "prompt": "How many orders are in the database?",
        "description": "Simple COUNT query"
    },
    {
        "level": "Easy",
        "prompt": "How many customers are there?",
        "description": "COUNT from customers table"
    },
    {
        "level": "Easy",
        "prompt": "How many products are in the catalog?",
        "description": "COUNT from products table"
    },
    {
        "level": "Easy",
        "prompt": "How many sellers are there?",
        "description": "COUNT from sellers table"
    },
    {
        "level": "Easy",
        "prompt": "What is the average order value?",
        "description": "AVG aggregation"
    },
    {
        "level": "Medium",
        "prompt": "What are the top 5 product categories by number of orders?",
        "description": "JOIN + GROUP BY + ORDER BY + LIMIT with translation"
    },
    {
        "level": "Medium",
        "prompt": "What are the most common payment methods?",
        "description": "GROUP BY payment type with counts"
    },
    {
        "level": "Medium",
        "prompt": "How many orders were placed in January 2017?",
        "description": "Date filtering with strftime"
    },
    {
        "level": "Medium",
        "prompt": "Which states have the most orders?",
        "description": "GROUP BY state with counts"
    },
    {
        "level": "Medium",
        "prompt": "What is the total revenue by payment method?",
        "description": "SUM aggregation with GROUP BY"
    },
    {
        "level": "Medium",
        "prompt": "Show me the monthly order trends in 2017",
        "description": "Date extraction and grouping"
    },
    {
        "level": "Medium",
        "prompt": "How many orders were delivered vs pending?",
        "description": "COUNT with WHERE condition on status"
    },
    {
        "level": "Advanced",
        "prompt": "Which sellers have the best customer ratings?",
        "description": "JOIN sellers with order_reviews through order_items and orders"
    },
    {
        "level": "Advanced",
        "prompt": "What is the average review score by product category?",
        "description": "JOIN products with order_reviews and category_translation"
    },
    {
        "level": "Advanced",
        "prompt": "Which states have the highest average order values?",
        "description": "JOIN orders with customers, AVG aggregation, ORDER BY DESC"
    },
    {
        "level": "Advanced",
        "prompt": "How much revenue did each seller generate?",
        "description": "SUM with seller joins to order_items and payments"
    },
    {
        "level": "Advanced",
        "prompt": "What is the average shipping cost by state?",
        "description": "JOIN with freight_value aggregation"
    },
    {
        "level": "Advanced",
        "prompt": "Which product categories have the most reviews?",
        "description": "COUNT with multiple JOINs and GROUP BY"
    },
    {
        "level": "Advanced",
        "prompt": "Show me the top 10 customers by total spending",
        "description": "GROUP BY customer with SUM and ORDER BY DESC LIMIT 10"
    },
    {
        "level": "Advanced",
        "prompt": "What percentage of orders were delivered on time?",
        "description": "COUNT with WHERE condition, date comparison"
    },
    {
        "level": "Advanced",
        "prompt": "Which payment methods are most popular by state?",
        "description": "Complex GROUP BY with multiple conditions"
    },
    {
        "level": "Advanced",
        "prompt": "Show me the order trends for Q4 2017",
        "description": "Date filtering with quarters"
    },
    {
        "level": "Advanced",
        "prompt": "How does average review score vary by payment method?",
        "description": "JOIN payment data with reviews"
    },
    {
        "level": "Advanced",
        "prompt": "What are the top product categories by total revenue in 2017?",
        "description": "Complex JOIN with translation and year filtering"
    },
    {
        "level": "Advanced",
        "prompt": "Which sellers have zero reviews?",
        "description": "LEFT JOIN to find null values"
    },
    {
        "level": "Advanced",
        "prompt": "Show me the customer count by state",
        "description": "DISTINCT COUNT with GROUP BY"
    },
    {
        "level": "Advanced",
        "prompt": "What is the average order value by customer state and payment method?",
        "description": "Multiple GROUP BY columns"
    },
    {
        "level": "Advanced",
        "prompt": "How many orders did the top 5 sellers make?",
        "description": "Subquery-like aggregation or multiple JOINs"
    },
    {
        "level": "Advanced",
        "prompt": "Show me the monthly revenue trends from January to June 2017",
        "description": "Date range with SUM aggregation"
    },
    {
        "level": "Advanced",
        "prompt": "Which product categories are most popular in each state?",
        "description": "Complex multi-table JOIN with GROUP BY multiple columns"
    },
]


def run_tests():
    """Run all test prompts and display results"""
    
    print("=" * 80)
    print("E-COMMERCE AI AGENT - COMPREHENSIVE TEST SUITE (30 Prompts)")
    print("=" * 80)
    print()
    
    results = {
        "Easy": {"passed": 0, "failed": 0, "total": 0},
        "Medium": {"passed": 0, "failed": 0, "total": 0},
        "Advanced": {"passed": 0, "failed": 0, "total": 0}
    }
    
    for i, test in enumerate(test_prompts, 1):
        level = test["level"]
        prompt = test["prompt"]
        description = test["description"]
        
        print(f"\n{'─' * 80}")
        print(f"Test {i:2d} | Level: {level:10s} | {description}")
        print(f"{'─' * 80}")
        print(f"Prompt: {prompt}")
        print()
        
        try:
            start_time = time.time()
            result = query_agent(prompt)
            elapsed_time = time.time() - start_time
            
            answer = result["answer"]
            
            if result["success"] and "Agent stopped due to iteration" not in answer and "iteration limit" not in answer.lower():
                print(f"✓ SUCCESS ({elapsed_time:.1f}s)")
                if len(answer) > 200:
                    print(f"Answer:\n{answer[:200]}...\n")
                else:
                    print(f"Answer:\n{answer}\n")

                # Check for visualization
                if result.get("figure") is not None:
                    print("📊 Visualization created")
                else:
                    print("📊 No visualization")

                results[level]["passed"] += 1
            else:
                print(f"✗ FAILED ({elapsed_time:.1f}s)")
                if "Agent stopped" in answer or "iteration limit" in answer.lower():
                    print(f"Error: Agent hit iteration limit\n")
                else:
                    print(f"Error: {result['answer'][:150]}\n")
                results[level]["failed"] += 1
                
        except Exception as e:
            print(f"✗ EXCEPTION: {str(e)[:150]}\n")
            results[level]["failed"] += 1
        
        results[level]["total"] += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_passed = 0
    total_failed = 0
    
    for level in ["Easy", "Medium", "Advanced"]:
        stats = results[level]
        passed = stats["passed"]
        failed = stats["failed"]
        total = stats["total"]
        pct = (passed / total * 100) if total > 0 else 0
        
        total_passed += passed
        total_failed += failed
        
        print(f"{level:10s}: {passed:2d}/{total:2d} passed ({pct:5.1f}%)")
    
    print("─" * 80)
    grand_total = total_passed + total_failed
    grand_pct = (total_passed / grand_total * 100) if grand_total > 0 else 0
    print(f"{'TOTAL':10s}: {total_passed:2d}/{grand_total:2d} passed ({grand_pct:5.1f}%)")
    print("=" * 80)


def test_followup_functionality():
    """Test the conversational follow-up functionality"""
    print("\n" + "=" * 80)
    print("FOLLOW-UP FUNCTIONALITY TEST")
    print("=" * 80)

    # Test initial query
    print("\n1. Initial Query:")
    result1 = query_agent_with_followup("What are the top 5 product categories by number of orders?")
    print(f"   Answer: {result1['answer'][:100]}...")
    print(f"   Follow-up available: {result1.get('follow_up_available', False)}")

    # Test follow-up 'yes'
    print("\n2. Follow-up 'Yes':")
    result2 = query_agent_with_followup("yes", is_followup=True)
    print(f"   Answer: {result2['answer'][:150]}...")
    print(f"   Additional visualization: {result2.get('figure') is not None}")

    # Test follow-up 'no'
    print("\n3. Follow-up 'No':")
    result3 = query_agent_with_followup("no", is_followup=True)
    print(f"   Answer: {result3['answer']}")

    print("\n✓ Follow-up functionality test completed")

if __name__ == "__main__":
    run_tests()
    test_followup_functionality()
