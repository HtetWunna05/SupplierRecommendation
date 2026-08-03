import pandas as pd

def recommend_best_supplier(category_name):
    # 1. Load the cleaned data
    try:
        df = pd.read_csv("data/cleaned_master_data.csv")
    except FileNotFoundError:
        return "❌ Error: Cleaned data not found. Please run Option 1 first."
    
    # 2. CASE-INSENSITIVE SEARCH SETUP
    # We convert the column to uppercase once
    df['part_number'] = df['part_number'].astype(str).str.upper()
    # We convert the user's input to uppercase
    search_query = str(category_name).strip().upper()

    # 3. Search using 'contains' (Partial matching)
    # This allows typing 'Elect' to find 'Electronics'
    options = df[df['part_number'].str.contains(search_query, na=False)].copy()

    if options.empty:
        # Suggest valid categories if search fails
        samples = df['part_number'].unique()[:5]
        return (f"\n❌ Category '{category_name}' not found.\n"
                f"💡 Valid suggestions: {', '.join(samples)}")

    # 4. RECOMMENDATION LOGIC
    # Score = (Reliability * 0.7) + (Consistency * 0.3)
    options['final_score'] = (options['reliability_score'] * 0.7) + (options['consistency_index'] * 0.3)
    
    # Find the single best entry
    winner = options.sort_values(by='final_score', ascending=False).iloc[0]

    # 5. Format Result
    return (
        f"\n--- 🏆 TOP RECOMMENDED SUPPLIER ---"
        f"\n📦 Category: {winner['part_number']}"
        f"\n🏢 Supplier ID: {winner['supplier']}"
        f"\n⭐ Reliability Score: {winner['reliability_score']}%"
        f"\n⏱️ Avg Delay: {winner['avg_lead_time']} days"
        f"\n📊 Overall Performance Score: {winner['final_score']:.2f}/100"
        f"\n----------------------------------"
    )

if __name__ == "__main__":
    # Test line
    cat = input("Enter Category:(TEXTILES, MACHINERY, FOOD, PHARMA, ELECTRONICS) ")
    print(recommend_best_supplier(cat))