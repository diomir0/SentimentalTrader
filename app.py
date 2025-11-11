from .sentimentalTrader import SentimentalTrader

if __name__ == "__main__":
    print("="*70)
    print("""SENTIMENTAL TRADER""")
    print("Features: Adversarial + Recursive + Temporal Decay")
    print("="*70)
    print("\nPrerequisites:")
    print("1. Install Ollama: curl https://ollama.ai/install.sh | sh")
    print("2. Pull Mistral: ollama pull mistral")
    print("3. Start Ollama: ollama serve")
    print("\nThis bot will:")
    print("  - Run Bull vs Bear debate for each decision")
    print("  - Recursively validate decisions")
    print("  - Track signal decay over time")
    print("  - Auto-trigger reassessment when signals get stale")
    print("="*70)
    print()
    
    user_mode = input("\nTrading Mode? (paper/live/analysis): ")
    while user_mode not in ['live', 'paper', 'analysis']:
        user_mode = input("\nPlease enter correct mode (paper/live/analysis): ")
    
    # Initialize bot
    bot = SentimentalTrader(initial_balance=400, mode=user_mode)
    
    # Configuration options
    print("Bot Configuration:")
    print(f"  Balance: ${bot.balance}")
    print(f"  Symbols: {bot.symbols}")
    print(f"  Adversarial: {bot.enable_adversarial}")
    print(f"  Recursive depth: {bot.recursive_depth}")
    print(f"  Signal half-life: {bot.signal_tracker.halflife_hours}h")
    print()
    
    # Run strategy
    try:
        user_input = input("Run strategy? (y/n): ").lower()
        if user_input == 'y':
            print("\n" + "="*70)
            print("STARTING STRATEGY EXECUTION")
            print("="*70 + "\n")
            
            bot.run_strategy()
            
            print("\n" + "="*70)
            print("STRATEGY EXECUTION COMPLETE")
            print("="*70 + "\n")
            
            # Show results
            bot.print_status()
            bot.analyze_performance()
            
            # Export data
            export = input("\nExport analysis to JSON? (y/n): ").lower()
            if export == 'y':
                bot.export_analysis()
                print("✓ Analysis exported to advanced_bot_analysis.json")
        
        else:
            print("\nBot initialized but not run. You can:")
            print("  - bot.run_strategy() - Run the strategy")
            print("  - bot.print_status() - Show current status")
            print("  - bot.get_comprehensive_decision('BTC-USD') - Test single decision")
            
    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
        bot.print_status()
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
