```
   ____         __  _                __       ________            __       
  / __/__ ___  / /_(_)_ _  ___ ___  / /____ _/ /_  __/______ ____/ /__ ____
 _\ \/ -_) _ \/ __/ /  ' \/ -_) _ \/ __/ _ `/ / / / / __/ _ `/ _  / -_) __/
/___/\__/_//_/\__/_/_/_/_/\__/_//_/\__/\_,_/_/ /_/ /_/  \_,_/\_,_/\__/_/   
``` 

# Intro
SentimentalTrader is a personal project aiming at creating a trading bot using sentiment analysis, adversarial LLM reasoning and temporal confidence decay. 

## Sentiment Analysis
While sentiment analysis used to be a reliable tool before the 2020s, using this method nowadays comes at high risk, with a performance decline of around 50% comparing pre-  ($\alpha = [0.05; 0.08]$)to post-2020 ($\alpha = [0.02; 0.04]$) periods. This decrease is mostly due to bot proliferation on X/Twitter (thanks Elon) and signal-to-noise ratio drop associated with higher manipulation of sentiment (on Reddit, for example). This is, especially true of cryptos, due to the bias of media coverage, their unregulated nature, the lack of enforcement against manipulation and the hardship of differenciating scams from legitimate projects.

In order to still make use of the sentiment information, the current project uses it mostly **defensively** instead of using it to find investment opportunities, and in the latter, rarer case, makes sure to check the market cap of the asset is large enough. Multiple sources (news, on-chain and market data) must align before taking action, the sentiment velocity is analysed before to detect pumps, and when manipulation is detected, an inverse strategy is used. 

## LLM Use 
First, let's clarify why using LLMs can be useful in our use case. LLM can provide *context-aware reasoning* instead of classical, mechanical averaging. This awareness allows adjustments of position sizing depending on your account's capital, modulation of weights attributed to different factors (e.g. technical vs news), information ambiguity handling, adaptation and self-improvement. Furthermore, the reasoning of the LLM can be made transparent in order to have a good understanding of the decisions it makes and on what grounds.

### Adversarial LLM System
In order to mitigate the inherent confirmation bias LLMs can display (which can have dramatic consequences when it comes to trade losses), we can use three LLMs:
- One bullish LLM arguing for buying.
- One bearish LLM arguing for selling and counters the bullish LLM's arguments.
- One judge LLM that weighs the arguments of the two above.
This configuration forces the consideration of countergarguments, thus enabling the judge LLM to have a full picture of the pros and cons of a given trade opportunity. This mimics how professional traders debate what positions to sell, buy or hold.

### Recursive validation
Additionally to the adversarial LLM system, the trading process also involves a recursive validation (with a default depth of 2) of the decision taken by the judge LLM in the step above, and either approves it, calls for revision or reverts it. The validation process stops once the recursive validation agrees with the decision fed to it. Ultimately, this step of the strategy:
- Catches overconfidence ("strong technical" gets challenged)
- Forces consideration of overlooked risks
- Acts like an internal devil's advocate
- Particularly valuable for small accounts that can't afford mistakes

## Temporal Confidence Decay
This concept stipulates that older signals (news, technical) have the potential to bias the estimation of the current market state, the latter changing day-to-day, hour-to-hour. The proposed fix is to decay the confidence of signals over time, meaning that older signals won't be trusted as much as more recent ones, thus forcing a reassessment of the situation.   

## Decision Flow
```
1. TEMPORAL CHECK
   ├─ Check signal age for existing positions
   ├─ If signals decayed → Force reassessment
   └─ Collect fresh data

2. DATA COLLECTION
   ├─ Technical indicators
   ├─ News sentiment
   ├─ Market structure
   └─ Generate new signals with timestamps

3. ADVERSARIAL ANALYSIS
   ├─ Bull LLM: Make case for buying
   ├─ Bear LLM: Counter with sell case
   └─ Judge LLM: Evaluate both arguments
   
4. RECURSIVE VALIDATION
   ├─ Critic LLM: Challenge the judge's decision
   ├─ If flaws found → Revise decision
   ├─ Challenge revised decision
   └─ Repeat until validated or depth reached

5. EXECUTION
   ├─ Check confidence threshold (≥0.5)
   ├─ Calculate position size
   ├─ Execute trade
   └─ Log everything for analysis
```
