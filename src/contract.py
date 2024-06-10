
def call_payoff(price, strike):
    return max(price-strike, 0)

def put_payoff(price, strike):
    return max(strike-price, 0)

class Contract:
    def __init__(self, strike, T, payoff, h=1):
        self.strike = strike
        self.T = T
        self.payoff = payoff
        self.h=h

    def get_payoff(self, price):
        return self.payoff(price, self.strike)