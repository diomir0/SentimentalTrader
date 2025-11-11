from .signal import Signal
from .temporalSignalTracker import TemporalSignalTracker
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import json
import logging
import requests
from textblob import TextBlob
import yfinance as yf
import ta
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentalTrader:
    '''
    Trading bot with:
    - Adversarial reasoning (bull vs. bear debate)
    - Recursive self-validation 
    - Temporal signal decay
    '''

    def __init__(self, initial_balance=400, llm_endpoint="http://localhost:11434", mode='paper'):
        """
        mode: 'paper' | 'live' | 'analysis'
        """
        self.mode = mode
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.virtual_positions = {}
        self.trade_history = []
        self.paper_trades = []
        self.decision_log = []

        # LLM configuration
        self.llm_endpoint = llm_endpoint
        self.llm_model = "dolphin3:latest"

        # Trading parameters 
        self.symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        self.max_position_size = 0.45
        self.stop_loss_pct = 0.04
        self.take_profit_pct = 0.10
        self.min_trade_size = 10

        # Temporal decay system
        self.signal_tracker = TemporalSignalTracker(
            halflife_hours=6.0, 
            max_age_hours=12.0
        )

        # Adversarial parameters
        self.enable_adversarial = True
        self.recursive_depth = 2        # Number of times decisions are challenged

        # Performance tracking
        self.reassessment_triggers = []
        
        logger.info(f"SentimentalTrader initialized in {mode.upper()} mode")
        logger.info(f"Features: Adversarial={self.enable_adversarial}, "
                    f"Recursive depth={self.recursive_depth}, "
                    f"Temporal decay half-life={self.signal_tracker.halflife_hours}h")
        self._verify_llm_connection()

    def _verify_llm_connection(self):
        '''Verify LLM is accessible'''
        try:
            response = requests.get(f"{self.llm_endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✓ LLM connected")
            else:
                logger.info("LLM not responding")
        except Exception as e:
            logger.error(f"Cannot connect to LLM: {e}")

    def query_llm(self, prompt: str, temperature: float = 0.1, role: str = "neutral") -> str:
        '''Query local LLM with role context'''
        try:
            # Add role context to prompt
            role_contexts = {
                'bull': "You are an AGGRESSIVELY BULLISH trader who looks for reasons to buy.",
                'bear': "You are an EXTREMELY BEARISH trader who looks for reasons to sell.",
                'judge': "You are an IMPARTIAL JUDGE who weighs evidence objectively.",
                'critic': "You are a CRITICAL ANALYST who finds flaws in reasoning.",
                'neutral': ""
            }

            full_prompt = role_contexts.get(role, "") + "\n\n" + prompt

            payload = {
                "model": self.llm_model,
                "prompt": full_prompt,
                "stream": False, 
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "max_tokens": 600
                }
            }

            response = requests.post(
                f"{self.llm_endpoint}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                logger.error(f"LLM error: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return ""

    #=============== DATA COLLECTION ===============#

    def collect_all_data(self, symbol: str) -> Dict:
        """Collect all data sources and create signals"""
        logger.info(f"Collecting data for {symbol}...")
        
        data = {
            'technical': self._collect_technical(symbol),
            'news': self._collect_news(symbol),
            'market': self._collect_market_structure(symbol),
            'timestamp': datetime.now()
        }
        
        # Generate and store signals
        self._generate_signals_from_data(symbol, data)
        
        return data
        
    def _collect_technical(self, symbol: str) -> Dict:
        '''Collect technical data'''
        try:
            df = yf.download(symbol, period="3mo", interval="1h", progress=False)
    
            if len(df) < 50:
                return {}
    
            # Calculate indicators
            df['SMA_7'] = ta.trend.sma_indicator(df['Close'].squeeze(), window=7)
            df['SMA_21'] = ta.trend.sma_indicator(df['Close'].squeeze(), window=21)
            df['RSI'] = ta.momentum.rsi(df['Close'].squeeze(), window=14)
    
            macd = ta.trend.MACD(df['Close'].squeeze())
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
    
            latest = df.iloc[-1]
            prev = df.iloc[-2]
    
            return {
                'price': float(latest['Close'].iloc[0]),
                'price_change_24h': float(latest['Close'].iloc[0] - df.iloc[-24]['Close'].iloc[0]),
                'rsi': float(latest['RSI'].iloc[0]),
                'sma_7': float(latest['SMA_7'].iloc[0]),
                'sma_21': float(latest['SMA_21'].iloc[0]),
                'macd': float(latest['MACD'].iloc[0]),
                'macd_signal': float(latest['MACD_signal'].iloc[0]),
                'trend': 'uptrend' if latest['SMA_7'].iloc[0] > latest['SMA_21'].iloc[0] else 'downtrend',
                'macd_crossover': 'bullish' if prev['MACD'].iloc[0] < prev['MACD_signal'].iloc[0] and latest['MACD'].iloc[0] > latest['MACD_signal'].iloc[0] else "bearish" if prev['MACD'].iloc[0] > prev['MACD_signal'].iloc[0] and latest['MACD'].iloc[0] < latest['MACD_signal'].iloc[0] else "none"
            }
    
        except Exception as e:
            logging.error(f"Technical data error: {e}")
            return {}

    def _collect_news(self, symbol: str) -> Dict:
        '''Collect news sentiment'''
        # Yahoo! finances API
        try:
            ticker = yf.Ticker(symbol)
            yf_news = ticker.news
    
            if not yf_news:
                return {'sentiment': 0.0, 'count': 0, 'headlines': []}
    
            yf_sentiments = []
            yf_headlines = []
    
            for item in yf_news[:10]:
                title = item['content'].get('title', '')
                if len(title) > 20: 
                    blob = TextBlob(title)
                    yf_sentiments.append(blob.sentiment.polarity)
                    yf_headlines.append(title)
    
            yf_avg_sentiment = np.mean(yf_sentiments) if yf_sentiments else 0.0
    
            return {
                'api': "yfinance",
                'sentiment': float(yf_avg_sentiment),
                'count': len(yf_headlines),
                'headlines': yf_headlines[:3]
            }
    
        except Exception as e:
            logger.error(f"YF News error: {e}")
            return {}
    
        '''
        # NewsAPI
        try: 
            newsapi = NewsApiClient(api_key=os.environ["NEWSAPI_KEY"])
            newsapi_news = newsapi.get_top_headlines(q='bitcoin',
                                                     sources='bbc-news,bloomberg,crypto-coins-news,financial-post,the-wall-street-journal,fortune',
                                                     category='business',
                                                     language='en',
                                                     country='us')  
            if not newsapi_news:
                return {'sentiment': 0.0, 'count': 0, 'headlines': []}
    
            newsapi_sentiments = []
            newsapi_headlines = []
    
            for item in newsapi_news[:10]:
                title = item.get('title', '')
                if len(title) > 20: 
                    blob = TextBlob(title)
                    newsapi_sentiments.append(blob.sentiment.polarity)
                    newsapi_headlines.append(title)
    
            newsapi_avg_sentiment = np.mean(newsapi_sentiments) if newsapi_sentiments else 0.0
    
            return {
                'api': "yf"
                'sentiment': float(newsapi_avg_sentiment),
                'count': len(newsapi_headlines)
                'headlines': newsapi_headlines[:3]
            }
    
        except Exception as e:
            logger.error(f"NewsAPI News error: {e}")
            return {}
        '''

    def _collect_market_structure(self, symbol: str) -> Dict:
        '''Collect market structure'''
        try: 
            df = yf.download(symbol, period="7d", interval="1h", progress=False)
    
            if len(df) < 24:
                return {}
    
            current = float(df.iloc[-1]['Close'].iloc[0])
            resistance = float(df['High'].squeeze().tail(168).nlargest(5).mean())
            support = float(df['Low'].squeeze().tail(168).nsmallest(5).mean())
    
            return {
                'support': support,
                'resistance': resistance,
                'distance_to_resistance': ((resistance - current) / current * 100),
                'distance_to_support': ((current - support) / current * 100)
            }
    
        except Exception as e:
            logger.error(f"Market structure error: {e}")
            return {}

    #=============== SIGNAL GENERATION ===============#
    
    def _generate_signals_from_data(self, symbol: str, data: Dict):
        '''Generate and store signals from collected data'''
        timestamp = data['timestamp']
    
        # Technical signals
        if data['technical']:
            tech = data['technical']
    
            # RSI signal
            if tech['rsi'] < 30:
                self.signal_tracker.add_signal(symbol, Signal(
                    signal_type='technical',
                    direction='bullish',
                    strength=0.8,
                    timestamp=timestamp,
                    details={'indicator': 'RSI', 'value': tech['rsi']}
                )) 
            elif tech['rsi'] > 70:
                self.signal_tracker.add_signal(symbol, Signal(
                    signal_type='technical',
                    direction='bearish',
                    strength=0.8,
                    timestamp=timestamp,
                    details={'indicator': 'RSI', 'value': tech['rsi']}
                ))
    
            # MACD signal
            if tech['macd_crossover'] == 'bullish':
                self.signal_tracker.add_signal(symbol, Signal(
                    signal_type='technical',
                    direction='bullish',
                    strength=0.8,
                    timestamp=timestamp,
                    details={'indicator': 'MACD', 'crossover': 'bullish'}
                )) 
            elif tech['macd_crossover'] == 'bearish':
                self.signal_tracker.add_signal(symbol, Signal(
                    signal_type='technical',
                    direction='bearish',
                    strength=0.8,
                    timestamp=timestamp,
                    details={'indicator': 'MACD', ' crossover': 'bearish'}
                ))
    
        # News signal
        if data['news'] and data['news']['count'] > 0:
            sentiment = data['news']['sentiment']
            strength = min(abs(sentiment), 1.0)
            direction = 'bullish' if sentiment > 0 else 'bearish' if sentiment < 0 else 'neutral'
    
            self.signal_tracker.add_signal(symbol, Signal(
                signal_type='news',
                direction=direction,
                strength=strength,
                timestamp=timestamp,
                details={'sentiment_score': sentiment, 'article_count': data['news']['count']}
            ))
    
        # Market structure signal
        if data['market']:
            market = data['market']
            if market['distance_to_support'] < 3:
                self.signal_tracker.add_signal(symbol, Signal(
                    signal_type='market_structure',
                    direction='bullish',
                    strength=0.6,
                    timestamp=timestamp,
                    details={'reason': 'near_support'}
                ))
            elif market['distance_to_resistance'] < 3:
                self.signal_tracker.add_signal(symbol, Signal(
                    signal_type='market_structure',
                    direction='bearish',
                    strength=0.6,
                    timestamp=timestamp,
                    details={'reason': 'near_resistance'}
                ))

    #=============== ADVERSARIAL REASONING  ===============#

    def adversarial_analysis(self, symbol: str, data: Dict) -> Dict:
        '''
        Bull vs Bear, judged by impartial LLM, if that exists
        '''
        logger.info(f"Starting adversarial analysis for {symbol}...")

        # Get temporal signal status
        signal_summary = self.signal_tracker.get_signal_summary(symbol)

        # Create data summary for LLMs
        data_summary = self._format_data_for_llm(symbol, data, signal_summary)

        # Step 1: Bull case
        logger.info("Querying bull LLM...")
        bull_prompt = f"""You are an AGGRESSIVELY BULLISH trader analyzing {symbol}.

        {data_summary}

        Make the STRONGEST possible case for BUYING. Find every bullish signal. Be agressive but honest, and DO NOT OVERESTIMATE the target confidence. Format your argument as:

        BULL CASE:
        - Key bullish factors: [list 3-5]
        - Why bearish factors are overblown: [address concerns]
        - Target confidence: [0.0-1.0]
        - Recommend position size: [percentage]
        """

        bull_case = self.query_llm(bull_prompt, temperature=0.2, role='bull')

        # Step 2: Bear case
        logger.info("Querying bear LLM...")
        bear_prompt = f"""You are an EXTREMELY BEARISH trader analyzing {symbol}.

        {data_summary}

        The bull trader argued:
        {bull_case}

        Make the STRONGEST possible case for SELLING or AVOIDING. Find every bearish signal and counter the bull's arguments. Be pessimistic but honest, and DO NOT OVERESTIMATE the risk. Format your argument as:

        BEAR CASE:
        - Key bearish factors: [list 3-5]
        - Why bullish factors are overblown: [counter bull's points]
        - Risk assessment: [What could go wrong]
        - Recommended action: [sell/avoid]
        """

        bear_case = self.query_llm(bear_prompt, temperature=0.2, role='bear')

        # Step 3: Judge evaluates both 
        logger.info("Querying judge LLM...")
        judge_prompt = f"""You are an IMPARTIAL JUDGE evaluating a trading debate for {symbol}.

        {data_summary}

        BULL ARGUMENT:
        {bull_case}

        BEAR ARGUMENT:
        {bear_case}

        Evaluate both arguments obkectively. Consider:
        1. Which side has more verifiable facts vs. speculation?
        2. Are the risks real or exaggerated?
        3. Are the opportunities real or overhyped?
        4. Given the account size (${self.balance:.2f}), what's the right decision?

        Provide your judgement in JSON format:
        {{
            "decision": "buy"|"sell"|"hold",
            "confidence": 0.0-1.0,
            "position_size_pct": 0-100,
            "reasoning": "explain which argument won and why",
            "bull_score": 0-10,
            "bear_score": 0-10,
            "key_deciding_factor": ["factor1", "factor2"]
        }}
        """

        judge_response = self.query_llm(judge_prompt, temperature=0.1, role='judge')

        # Parse judge's decision
        try:
            json_start = judge_response.find('{')
            json_end = judge_response.find('}') + 1
            judge_decision = json.loads(judge_response[json_start:json_end])

            # Add metadata
            judge_decision['bull_case'] = bull_case[:200]
            judge_decision['bear_case'] = bear_case[:200]
            judge_decision['adversarial_used'] = True

            return judge_decision

        except Exception as e:
            logger.error(f"Failed to parse judge decision: {e}")
            return None

    #=============== RECURSIVE VALIDATION  ===============#

    def recursive_validation(self, symbol: str, initial_decision: Dict, depth: int = 2) -> Dict:
        """
        Recursively challenge and validate the decision
        """
        logger.info(f"Starting recursive validation (depth={depth})...")

        current_decision = initial_decision
        validation_history = []

        for level in range(depth):
            logger.info(f"Validation level {level + 1}/{depth}...")

            critique_prompt = f"""You are a CRITICAL ANALYST reviewing a trading decision for {symbol}.

            PROPOSED DECISION:
            - Action: {current_decision['decision']}
            - Confidence: {current_decision['confidence']}
            - Position size: {current_decision.get('position_size_pct', 0)}%
            - Reasoning: {current_decision['reasoning']}
            
            CHALLENGE THIS DECISION:
            1. What's the WEAKEST part of this reasoning?
            2. What evidence CONTRADICTS this decision?
            3. What risks are being UNDERESTIMATED?
            4. If this decision loses money, why would it be?
            5. What would a more experienced trader critique?

            After this critique, respond in JSON:
            {{
                "verdict": "validated"|"needs_revision"|"reverse_decision",
                "critique": "detailed critisism",
                "revised_decision": "buy"|"sell"|"hold" (if needs revision),
                "revised confidence": 0.0-1.0 (if needs revision),
                "revised_reasoning": "explanantion" (if needs revision)
            }}
            """

            critique_response = self.query_llm(critique_prompt, temperature=0.1, role='critic')
            
            try:
                json_start = critique_response.find('{')
                json_end = critique_response.find('}') + 1
                critique = json.loads(critique_response[json_start:json_end])

                validation_history.append({
                    'level': level + 1,
                    'verdict': critique['verdict'],
                    'critique': critique['critique']
                })

                if critique['verdict'] == 'validated':
                    logger.info(f"✓ Decision validated at level {level + 1}")
                    break
                elif critique['verdict'] in ['needs_revision', 'reverse_decision']:
                    logger.info(f"⚠ Decision revised at level {level + 1}")
                    current_decision['decision'] = critique.get('revised_decision', current_decision['decision'])
                    current_decision['confidence'] = critique.get('revised_confidence', current_decision['confidence'])
                    current_decision['reasoning'] = critique.get('revised_reasoning', current_decision['reasoning'])

            except Exception as e:
                logger.error(f"Validation level {level + 1} failed: {e}")
                break

        current_decision['validation_history'] = validation_history
        current_decision['validation_depth'] = len(validation_history)

        return current_decision

    #=============== MAIN DECISION PIPELINE  ===============#

    def get_comprehensive_decision(self, symbol: str) -> Optional[Dict]:
        """
        Complete decision pipeline:
        1. Check temporal decay (should we reassess?)
        2. Collect fresh data
        3. Adversarial analysis
        4. Recursive validation
        5. Final decision
        """
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE ANALYSIS FOR {symbol}")
        print(f"{'='*70}")

        # Step 1: Check if we need to reassess existing position
        if symbol in self.positions:
            should_reassess, reason = self.signal_tracker.should_reassess(symbol)
            if should_reassess:
                logger.warning(f"⚠ Position reassessment triggered: {reason}")
                self.reassessment_triggers.append({
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'reason': reason
                })
                
        # Step 2: Collect all data
        data = self.collect_all_data(symbol)

        if not data['technical']:
            logger.error(f"Cannot analyze {symbol} - insufficient data")
            return None

        # Step 3: Adversarial analysis
        if self.enable_adversarial:
            decision = self.adversarial_analysis(symbol, data)
        else:
            decision = self._simple_analysis(symbol, data)

        if not decision:
            logger.error(f"Failed to generate decision for {symbol}")
            return None

        # Step 4: Recursive validation
        if self.recursive_depth > 0:
            decision = self.recursive_validation(symbol, decision, self.recursive_depth)

        # Step 5: Add metadata
        decision['symbol'] = symbol
        decision['timestamp'] = datetime.now().isoformat()
        decision['temporal_confidence'] = self.signal_tracker.get_current_confidence(symbol)
        decision['data'] = data

        # Log decision
        self.decision_log.append(decision)

        print(f"\n{'='*70}")
        print(f"FINAL DECISION FOR {symbol}")
        print(f"{'='*70}")
        print(f"Action: {decision['decision'].upper()}")
        print(f"Confidence: {decision['confidence']:.2f}")
        print(f"Position Size: {decision.get('position_size_pct', 0)}%")
        print(f"Reasoning: {decision['reasoning'][:150]}...")
        if decision.get('validation_history'):
            print(f"Validation: Passed {len(decision.get('validation_history'))} levels")
        print(f"{'='*70}\n")

        return decision

    def _simple_analysis(self, symbol: str, data: Dict) -> Dict:
        """Fallback simple analysis if adversarial is disabled"""
        return {
            'decision': 'hold',
            'confidence': 0.5,
            'position_size_pct': 0,
            'reasoning': 'Simple ananysis - adversarial disbaled'
        }

    def _format_data_for_llm(self, symbol: str, data: Dict, signal_summary: str) -> str:
        """Format collected data for LLM consumption"""
        tech = data.get('technical', {})
        news = data.get('news', {})
        market = data.get('market', {})

        # Position info
        position_info = "No current position"
        if symbol in self.positions:
            pos = self.positions[symbol]
            position_info = f"Current position: {pos['shares']:.4f} shares at ${pos['avg_price']:.2f}"

            return f"""
            SYMBOL: {symbol}
            ACCOUNT BALANCE: ${self.balance:.2f}
            {position_info}

            TEMPORAL SIGNAL STATUS:
            {signal_summary}

            TECHNICAL DATA:
            - PRICE: ${tech.get('price', 0):.2f}
            - 24h Change: {tech.get('price_change_24h', 0):+.2f}%
            - RSI: {tech.get('rsi', 50):.1f}
            - Trend: {tech.get('trend', 'unknown')}
            - MACD: {tech.get('macd_crossover', 'none')}

            NEWS SENTIMENT:
            - Sentiment: {news.get('sentiment', 0):+.2f}
            - Articles: {news.get('count', 0)}
            - Headlines: {'; '.join(news.get('headlines', [])[:2])}

            MARKET STRUCTURE:
            - Support: ${marker.get('support', 0):.2f}
            - Resistance: ${market.get('resistance', 0):.2f}
            - Distance to resistance: {market.get('distance_to_resistance', 0):.1f}%
            - Distance to support: {market.get('distance_to_support', 0):.1f}%
            """

    #=============== EXECUTION  ===============#

    def execute_decision(self, decision: Dict) -> bool:
        """Execute the final decision"""
        symbol = decision['symbol']
        action = decision['decision']
        confidence = decision['confidence']

        if self.mode == 'analysis':
            self._log_analysis(decision)
            return False

        elif self.mode == 'paper':
            print("Running in paper trading mode...")

        else: 
            print("Running in live trading mode...")

        # Confidence threshold
        if confidence < 0.5:
            logger.info(f"Skipping {action} - confidence too low ({confidence:.2f})")
            return False

        current_price = decision['data']['technical']['price']

        if action == 'buy':
            if symbol in self.positions:
                logger.info(f"Already holding {symbol}")
                return False

            # Calculate position size
            max_investment = self.balance * self.max_position_size
            actual_investment = max_investment * (decision.get('position_size_pct', 50) / 100)
            actual_investment = min(actual_investment, self.balance * 0.95)

            if actual_investment < self.min_trade_size:
                logger.info(f"Investment too small: ${actual_investment:.2f}")
                return False
    
            shares = actual_investment / current_price
            self.balance -= actual_investment
            self.positions[symbol] = {
                'shares': shares,
                'avg_price': current_price,
                'entry_time': datetime.now(),
                'entry_reasoning': action['reasoning'][:200]
            }
    
            self.trade_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'buy',
                'shares': shares,
                'price': current_price,
                'total': actual_investment,
                'confidence': confidence,
                'adversarial': decision.get('adversarial_used', False),
                'validated': decision.get('validation_depth', 0) > 0
            })
    
            logger.info(f"✓ EXECUTED BUY: {shares:.4f} {symbol} @ ${current_price:.2f}")
            return True

        elif action == 'sell':
            if symbol not in self.positions:
                logger.info(f"No position in {symbol} to sell")
                return False

            pos = self.positions[symbol]
            shares = pos['shares']
            revenue = shares * current_price
            cost = shares * pos['avg_price']
            profit = revenue - cost
            profit_pct = (profit / cost * 100)
    
            self.balance += revenue
            del self.positions[symbol]
    
            # Clear old signals after selling
            self.signal_tracker.signals_by_symbol[symbol] = []
    
            self.trade_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'sell',
                'shares': shares,
                'price': current_price,
                'total': revenue,
                'profit': profit,
                'profit_pct': profit_pct,
                'confidence': confidence,
                'adversarial': decision.get('adversarial_used', False),
                'validated': decison.get('validation_depth', 0) > 0
            })
    
            logger.info(f"✓ EXECUTED SELL: {shares:.4f} {symbol} @ ${current_price:.2f}"
                       f"(P/L: ${profit:+.2f} / {profit_pct:+.2f}%")
            return True

        else:    # Hold
            logger.info(f"HOLD {symbol} - no action taken")
            return False

    def _log_analysis(self, decision: Dict):
        """Log what would happen in analysis mode"""
        print(f"\n{'='*60}")
        print(f"📊 ANALYSIS-ONLY MODE - NO EXECUTION")
        print(f"{'='*60}")
        print(f"Symbol: {decision['symbol']}")
        print(f"Decision: {decision['decision'].upper()}")
        print(f"Confidence: {decision['confidence']:.2f}")
        print(f"Position Size: {decision.get('position_size_pct', 0)}%")
        print(f"Reasoning: {decision['reasoning']}")
        
        if decision.get('validation_history'):
            logger.info(f"\nValidation:")
            for v in decision['validation_history']:
                print(f"  Level {v['level']}: {v['verdict']}")
        
        logger.info(f"{'='*60}\n")

    def run_strategy(self):
        """Run complete trading strategy with all its features"""
        print(f"\n{'='*70}")
        print("RUNNING TRADING STRATEGY")
        print('='*70)
        print(f"""Features active:
    - Adversarial reasoning: {self.enable_adversarial}
    - Recursive validation depth: {self.recursive_depth}
    - Temporal decay half-life: {self.signal_tracker.halflife_hours}h""")
        print(f"{'='*70}\n")

        for symbol in self.symbols:
            try:
                # Get comprehensive decision
                decision = self.get_comprehensive_decision(symbol)

                if decision:
                    # Execute decision
                    self.execute_decision

                # Brief pause between symbols
                time.sleep(2)
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                import traceback 
                traceback.print_exc()

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        portfolio_value = self.balance

        for symbol, pos in self.positions.items():
            try:
                data = yf.download(symbol, period="1d", interval="1h", progress=False)
                if len(data) > 0:
                    current_price = data['Close'].iloc[-1]
                    portfolio_value += pos['shares'] * current_price
            except Exception as e:
                logger.error(f"Error valuing {symbol}: {e}")

        return portfolio_value

    def print_status(self):
        """Print comprehensive protfolio status"""
        portfolio_value = self.get_portfolio_value()
        total_return = ((portfolio_value - self.initial_balance) / self.initial_balance * 100)

        print(f"\n{'='*70}")
        print("PORTFOLIO STATUS")
        print(f"{'='*70}")
        print(f"Portfolio Value: {portfolio_value:.2f}")
        print(f"Cash Balance: ${self.balance:.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total Trades: {len(self.trade_history)}")
        print(f"Decisions Made: {len(self.decision_log)}")
        print(f"Reassessment Triggered: {len(self.reassessment_triggers)}")

        # Calculate advanced metrics
        if self.decision_log:
            adversarial_decisions = sum(1 for d in self.decision_log if d.get('adversarial_used'))
            validated_decisions = sum(1  for d in self.decision_log if d.get('validation_depth', 0) > 0)
            avg_confidence = np.mean([d['confidence'] for d in self.decision_log])
            print(f"""\nDecision Quality:
            \t - Adversarial analysis: {adversarial_decisions}/{len(self.decision_log)}
            \t - Recursively validated: {validated_decisions}/{len(self.decision_log)}
            \t - Average confidence: {avg_confidence:.2f}
            """)

        # Current positions
        if self.positions:
            print(f"\nCurrent positions:")
            for symbol, pos in self.positions.items():
                try:
                    data = yf.download(symbol, period="1d", interval="1h", progress=False)
                    if len(data) > 0:
                        current_price = data['Close'].iloc[-1]
                        position_value = os['shares'] * current_price
                        pnl = ((current_price - pos['avg_price']) / pos['avg_price'] * 100)
                        hold_time = (datetime.now() - pos['entry_time']).total_seconds() / 3600
                        print(f"""\n{symbol}:
                        \tShares: {pos['shares']:.4f}
                        \tEntry: ${pos['avg_price']:.2f}
                        \tCurrent: ${current_price:.2f}
                        \tValue: ${position_value:.2f}
                        \tP/L: {pnl:+.2f}%
                        \tHold time: {hold_time:.1f}h
                        """)

                        # Check temporal decay
                        confidence = self.signal_tracker.get_current_confidence(symbol)
                        print(f"\tSignal confidence: {confidence['total']:.2f}")
                        
                        should_reassess, reason = self.signal_tracker.should_reassess(symbol)
                        if should_reassess:
                            print(f"\t⚠ WARNING: {reason}")

                except Exception as e:
                    print(f"\t{symbol}: Error - {e}")
        else:
            print("\nNo open positions")

         # Recent trades
        if self.trade_history:
            print(f"\n{'Recent Trades (last 5):'}")
            for trade in self.trade_history[-5:]:
                ts = trade['timestamp'].strftime('%m-%d %H:%M')
                action = trade['action'].upper()
                conf = trade.get('confidence', 0)
                adv = "✓" if trade.get('adversarial') else "✗"
                val = "✓" if trade.get('validated') else "✗"
                
                if 'profit' in trade:
                    print(f"  {ts} | {action} {trade['symbol']} @ ${trade['price']:.2f} | "
                          f"P/L: ${trade['profit']:+.2f} ({trade['profit_pct']:+.1f}%) | "
                          f"Conf: {conf:.2f} | Adv:{adv} Val:{val}")
                else:
                    print(f"  {ts} | {action} {trade['symbol']} @ ${trade['price']:.2f} | "
                          f"Conf: {conf:.2f} | Adv:{adv} Val:{val}")
        
        # Temporal decay warnings
        if self.reassessment_triggers:
            print(f"\n{'Recent Reassessment Triggers:'}")
            for trigger in self.reassessment_triggers[-3:]:
                ts = trigger['timestamp'].strftime('%m-%d %H:%M')
                print(f"  {ts} | {trigger['symbol']} | {trigger['reason']}")
        
        print(f"{'='*70}\n")
    
    def export_analysis(self, filename='advanced_bot_analysis.json'):
        """Export complete analysis including temporal data"""
        export_data = {
            'portfolio': {
                'balance': self.balance,
                'initial_balance': self.initial_balance,
                'portfolio_value': self.get_portfolio_value(),
                'return_pct': ((self.get_portfolio_value() - self.initial_balance) / 
                              self.initial_balance * 100)
            },
            'positions': self.positions,
            'decisions': self.decision_log,
            'trades': self.trade_history,
            'reassessments': self.reassessment_triggers,
            'temporal_signals': {
                symbol: [
                    {
                        'type': s.signal_type,
                        'direction': s.direction,
                        'strength': s.strength,
                        'decayed_strength': s.get_decayed_strength(self.signal_tracker.half_life_hours),
                        'age_hours': (datetime.now() - s.timestamp).total_seconds() / 3600,
                        'details': s.details
                    }
                    for s in signals
                ]
                for symbol, signals in self.signal_tracker.signals_by_symbol.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported analysis to {filename}")
    
    def analyze_performance(self):
        """Analyze bot performance and decision quality"""
        print(f"\n{'='*70}")
        print(f"PERFORMANCE ANALYSIS")
        print(f"{'='*70}")
        
        if not self.trade_history:
            print("No trades to analyze")
            return
        
        # Win rate
        closed_trades = [t for t in self.trade_history if 'profit' in t]
        if closed_trades:
            wins = [t for t in closed_trades if t['profit'] > 0]
            win_rate = len(wins) / len(closed_trades) * 100
            avg_win = np.mean([t['profit_pct'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['profit_pct'] for t in closed_trades if t['profit'] < 0])
            
            print(f"\nTrade Performance:")
            print(f"  Win rate: {win_rate:.1f}%")
            print(f"  Average win: {avg_win:+.2f}%")
            print(f"  Average loss: {avg_loss:+.2f}%")
            print(f"  Win/Loss ratio: {abs(avg_win/avg_loss) if avg_loss != 0 else 0:.2f}")
        
        # Adversarial vs non-adversarial
        adv_trades = [t for t in closed_trades if t.get('adversarial')]
        non_adv_trades = [t for t in closed_trades if not t.get('adversarial')]
        
        if adv_trades and non_adv_trades:
            adv_win_rate = len([t for t in adv_trades if t['profit'] > 0]) / len(adv_trades) * 100
            non_adv_win_rate = len([t for t in non_adv_trades if t['profit'] > 0]) / len(non_adv_trades) * 100
            
            print(f"\nAdversarial Impact:")
            print(f"  With adversarial: {adv_win_rate:.1f}% win rate")
            print(f"  Without adversarial: {non_adv_win_rate:.1f}% win rate")
            print(f"  Improvement: {adv_win_rate - non_adv_win_rate:+.1f}%")
        
        # Validated vs non-validated
        val_trades = [t for t in closed_trades if t.get('validated')]
        non_val_trades = [t for t in closed_trades if not t.get('validated')]
        
        if val_trades and non_val_trades:
            val_win_rate = len([t for t in val_trades if t['profit'] > 0]) / len(val_trades) * 100
            non_val_win_rate = len([t for t in non_val_trades if t['profit'] > 0]) / len(non_val_trades) * 100
            
            print(f"\nValidation Impact:")
            print(f"  With validation: {val_win_rate:.1f}% win rate")
            print(f"  Without validation: {non_val_win_rate:.1f}% win rate")
            print(f"  Improvement: {val_win_rate - non_val_win_rate:+.1f}%")
        
        # Temporal reassessments
        if self.reassessment_triggers:
            print(f"\nTemporal Decay Analysis:")
            print(f"  Reassessments triggered: {len(self.reassessment_triggers)}")
            
            # Check if reassessments prevented losses
            reassessed_symbols = set(r['symbol'] for r in self.reassessment_triggers)
            reassessed_trades = [t for t in closed_trades if t['symbol'] in reassessed_symbols]
            
            if reassessed_trades:
                reassessed_avg_return = np.mean([t['profit_pct'] for t in reassessed_trades])
                print(f"  Average return on reassessed positions: {reassessed_avg_return:+.2f}%")
        
        print(f"{'='*70}\n")

    def run_paper_trading_session(self, duration_days=7):
        """
        Run paper trading for specified duration
        Track hypothetical P&L
        """
        logger.info(f"Starting {duration_days}-day paper trading session")

        start_balance = self.balance

        for day in range(duration_days):
            logger.info(f"\n{'='*60}")
            logger.info(f"DAY {day + 1}/{duration_days}")
            logger.info(f"{'='*60}")

            # Run strategy
            self.run_strategy()

            # Show hypothetical performance
            self._show_paper_performance()

            # Wait 24 hours (or adjust for testing)
            if day < duration_days - 1:
                logger.info("Waiting 24 hours for next iteration...")
                #time.sleep(86400)

        logger.info(f"\n{'='*60}")
        logger.info(f"PAPER TRADING SESSION COMPLETE")
        logger.info(f"\n{'='*60}")
        final_balance = self.get_portfolio_value()
        total_return = ((final_balance - start_balance) / start_balance * 100)

        logger.info(f"Starting balance: ${start_balance:.2f}")
        logger.info(f"Ending balance: ${final_balance:.2f}")
        logger.info(f"Total return: {total_return:+.2f}%")
        logger.info(f"Total decisions: {len(self.decision_log)}")
        logger.info(f"Trades executed (paper): {len(self.paper_trades)}")

    def _show_paper_performance(self):
        """Show current paper trading performance"""
        print(f"\nPaper trading performance")
        print(f"\tVirtual balance: ${self.balance:.2f}")
        print(f"\tVirtual positions: {len(self.virtual_positions)}")

        if self.paper_trades:
            recent = self.paper_trades[-3:]
            print(f"\t Recent paper trades:")
            for trade in recent:
                action = trade['action'].upper()
                print(f"\t\t- {action} {trade['shares']:.4f} {trade['symbol']} @ ${trade['price']:.2f}")