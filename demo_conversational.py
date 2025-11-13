"""
Demo script showing the conversational follow-up functionality
"""

from agent import query_agent_with_followup

def demo_conversation():
    print("🤖 E-Commerce AI Agent - Conversational Demo")
    print("=" * 50)

    # Simulate a conversation
    print("\nUser: What are the top 5 product categories by number of orders?")
    print("\nAgent:")

    # Initial query
    result1 = query_agent_with_followup("What are the top 5 product categories by number of orders?")
    print(result1['answer'])
    if result1.get('figure'):
        print("📊 [Visualization created]")

    # User wants more info
    print("\nUser: yes")
    print("\nAgent:")

    result2 = query_agent_with_followup("yes", is_followup=True)
    print(result2['answer'])
    if result2.get('figure'):
        print("📊 [Additional visualization created]")

    # User is done
    print("\nUser: no")
    print("\nAgent:")

    result3 = query_agent_with_followup("no", is_followup=True)
    print(result3['answer'])

    print("\n" + "=" * 50)
    print("✅ Conversational demo completed!")

if __name__ == "__main__":
    demo_conversation()