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

## Adversarial LLM Use 
First, let's clarify why using LLMs can be useful in our use case. LLM can provide *context-aware reasoning* instead of classical, mechanical averaging. This awareness allows adjustments of position sizing depending on your account's capital, modulation of weights attributed to different factors (e.g. technical vs news), information ambiguity handling, adaptation and self-improvement. Furthermore, the reasoning of the LLM can be made transparent in order to have a good understanding of the decisions it makes and on what grounds.

### Adversarial LLM System

