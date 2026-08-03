import clean_data
import database
import recommend

def main_menu():
    while True:
        print("\n" + "="*40)
        print("🛡️  US SUPPLY CHAIN RISK SYSTEM")
        print("="*40)
        print("1. 🚀 Clean Data & Update Risk Metrics")
        print("2. 🗄️  Sync to SQL Database")
        print("3. 🏆 Get Supplier Recommendation")
        print("4. 🚪 Exit")
        
        # .strip() handles accidental spaces
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            clean_data.run_cleaning()
        elif choice == '2':
            database.create_database()
        elif choice == '3':
            while True:
                print("\n" + "-"*30)
                print("🔍 RECOMMENDATION SEARCH")
                print("(Type 'back' to return to Main Menu)")
                category = input("Enter Product Category:(TEXTILES, MACHINERY, FOOD, PHARMA, ELECTRONICS) ").strip()
                
                # CASE-INSENSITIVE BACK CHECK
                if category.lower() == 'back':
                    break 
                
                if category == "":
                    print("⚠️ Please enter a category or type 'back'.")
                    continue
                    
                print(recommend.recommend_best_supplier(category))
                
        elif choice == '4':
            print("Exiting System... Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main_menu()