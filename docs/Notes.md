Thompson:
	The agent should use a binomial distribution to determine if they will be informed. This agents demand is calculated using draws from multiple distributions. Their bid-ask prices is a confidence interval on their prior distribution over the value of the asset, where $alpha$ is their risk-aversion.
	 The bid-ask quantities are the results of a draw from negative-binomial distributions, where $r=1$ and it is interpreted as how many shares should I buy before there is one that I shouldn't buy. The probability parameter should be tied to price in some way. Perhaps $X*price$ for ask-quantity and $1-X*price$ for bid-quantity where $X$ is the value being tweaked by the learning algorithm.

	Probability of success for geometric = p-value for price in prior distribution of asset values.
		

	Should Neural Networks and GA share the informed approach with Thompson?

**Does giving the informed agents more repetitions than the uninformed agents allow prices to be locally constructive and have costly information?**
- Cost function that scales cost as extra repetitions are purchased
- 


### **Dynamically Adjust Information Costs:**

- Make the **cost function endogenous** based on:
    
    - Number of informed traders (crowding effect).
        
    - Market volatility (when prices are noisy, information is more valuable → higher willingness to pay).
        

|Metric|Interpretation|
|---|---|
|**Entropy of agents' beliefs**|How diverse/uncertain are subjective priors/posteriors?|
|**Mutual information between agents' trades & prices**|Degree to which agents’ behavior affects price discovery (i.e., reflexivity).|
|**Effective sample size (ESS)** over generations|How concentrated or dispersed are agents’ posterior beliefs?|
|**Volatility clustering emerging endogenously**|Can mimic real-world stylized facts without needing exogenous shocks?|

**Could even model this as:**

$C_{info} = f(\sigma_{price}, \text{\# of informed agents})$

should the model learn optimal trades and then estimate the model that performs the best trades.

