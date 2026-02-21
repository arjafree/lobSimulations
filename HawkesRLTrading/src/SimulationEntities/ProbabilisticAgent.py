from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent
from HawkesRLTrading.src.SimulationEntities.ICRLAgent import get_queue_priority
from typing import Dict, Optional, Any
import numpy as np

class ProbabilisticAgent(GymTradingAgent):
    def __init__(self, seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random",
                 Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5,
                 wake_on_MO: bool=True, wake_on_Spread: bool=True, cashlimit=1000000,inventorylimit=100,
                 rewardpenalty = 0.1, transaction_cost=0, start_trading_lag=0):
        """
        Deals w probabilities of next MO
        """
        super().__init__(seed=seed, log_events=log_events, log_to_file=log_to_file, strategy=strategy,
                         Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO,
                         wake_on_Spread=wake_on_Spread, cashlimit=cashlimit, inventorylimit=inventorylimit, start_trading_lag=start_trading_lag,
                         truncation_enabled=False)

        self.resetseed(seed)

        self.rewardpenalty = rewardpenalty  # inventory penalty
        self.last_state, self.last_action = None, None
        self.inv_threshold = 10
        self.avgPrices = (0,0)
        self.init_cash = self.cash
        self.cols= ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                    "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
        self.trajectory_buffer = []
        self.transaction_cost = transaction_cost

    def readData(self, data):
        """
        Read and preprocess trading data

        :param data: Input trading data
        :return: Processed state tensor
        """
        p_a, q_a = data['LOB0']['Ask_L1']
        p_b, q_b = data['LOB0']['Bid_L1']
        self.mid = 0.5*(p_a + p_b)
        _, qD_a = data['LOB0']['Ask_L2']
        _, qD_b = data['LOB0']['Bid_L2']
        pos = data['Positions']
        n_as = get_queue_priority(data, pos, 'Ask_L1')
        n_as = np.append(n_as, get_queue_priority(data, pos, 'Ask_L2'))
        n_as = np.append(n_as, [q_a+qD_a])
        n_bs = get_queue_priority(data, pos, 'Bid_L1')
        n_bs = np.append(n_bs, get_queue_priority(data, pos, 'Bid_L2'))
        n_bs = np.append(n_bs, [q_b+qD_b])
        n_a, n_b = np.min(n_as), np.min(n_bs)
        lambdas = data['current_intensity']
        past_times = data['past_times']
        state = [[self.cash, self.Inventory['INTC'], p_a, p_b, q_a, q_b, qD_a, qD_b, n_a, n_b, (p_a + p_b)*0.5] , lambdas.flatten(), past_times.flatten()]
        return state
    def store_transition(self, ep, state, action, reward, next_state, done):
        """
        Store transition in trajectory buffer

        :param state: Current state
        :param action: Chosen action (tuple of d and u)
        :param reward: Received reward
        :param next_state: Next state
        :param done: Episode termination flag
        """
        # Unpack action components
        # d, u = action

        # Store additional trajectory information
        transition = (
            state,       # Current state
            action[0],           # Decision action
            action[1:],           # Utility action
            reward,      # Reward
            next_state,  # Next state
            int(done)         # Done flag
        )
        self.trajectory_buffer.append((ep, transition))
    def get_action(self, data, epsilon=0.1):
        """
        Epsilon-greedy policy to balance exploitation with exploration.

        Args:
            data: The input data
            epsilon: Probability of choosing a random action (default: 0.1)

        Returns:
            action, size, logits_d, logits_u
        """
        if self.breach:
            mo = 4 if self.countInventory() > 0 else 7
            return ((mo, 1),(12,1))
        size = 1
        origData = data.copy()
        data = self.readData(data)

        # Store current state for learning
        self.last_state = data

        lambdas = data[1]
        lambdas_norm = lambdas/np.sum(lambdas)
        skew = (data[0][-3] - data[0][-2])/(0.5*(data[0][4] + data[0][5])) # (na - nb) / mean(qa, qb)
        inv = data[0][1]
        pa, pb = data[0][2], data[0][3]
        actions = []
        if np.argmax(lambdas_norm) == 4: # mo ask - hits ask
            if inv < -self.inv_threshold:
                # if lambdas[7] + 1000 < lambdas[4]:
                    return ((12,size),(7,size)) # mo_bid
            elif (inv > self.inv_threshold) or ( (inv > 0)and(pa > (self.cash-self.init_cash)/inv) ): # goodfill
                if data[0][-3]/data[0][4] > 1:
                    actions.append((9,size)) # lo bid top
                if data[0][-2]/data[0][5]<1:
                    actions.append((2,size))
            elif ( (inv > 0)and(pa < (self.cash-self.init_cash)/inv) ): #badfill
                if len(origData['Positions']['Ask_L1']) != 0: actions.append((3, size))
                if data[0][-3]/data[0][4] > 1:
                    actions.append((9,size)) # lo bid top
                if data[0][-2]/data[0][5]<1:
                    actions.append((3,size))
        elif np.argmax(lambdas_norm) == 7: # mo bid - hits bid
            if inv > self.inv_threshold:
                # if lambdas[4] + 1000 < lambdas[7]:
                    return ((12,size),(4,size)) # mo_ask
            elif (inv < -self.inv_threshold) or ( (inv < 0)and(pb < (self.cash-self.init_cash)/inv) ): # goodfill
                if data[0][-3]/data[0][4] < 1:
                    actions.append((9,size)) # lo bid top
                if data[0][-2]/data[0][5]>1:
                    actions.append((2,size))
            elif ( (inv < 0)and(pb > (self.cash-self.init_cash)/inv) ): #badfill
                if len(origData['Positions']['Bid_L1']) != 0: actions.append((8, size))
                if data[0][-3]/data[0][4] < 1:
                    actions.append((9,size)) # lo bid top
                if data[0][-2]/data[0][5] > 1:
                    actions.append((2,size))
        else:
            if skew > 0.5:
                if (data[0][4] - data[0][-2])/(0.5*(data[0][4] + data[0][5])) < skew:
                    if len(origData['Positions']['Ask_L2']) != 0: actions.append((1,size))
                    actions.append((2,size))
                if (data[0][-3] - data[0][5])/(0.5*(data[0][4] + data[0][5])) < skew:
                    if len(origData['Positions']['Bid_L2']) != 0: actions.append((11,size))
                    actions.append((9,size))
            if skew < -0.5:
                if (data[0][4] - data[0][-2])/(0.5*(data[0][4] + data[0][5])) > skew:
                    if len(origData['Positions']['Ask_L2']) != 0: actions.append((1,size))
                    actions.append((2,size))
                if (data[0][-3] - data[0][5])/(0.5*(data[0][4] + data[0][5])) > skew:
                    if len(origData['Positions']['Bid_L2']) != 0: actions.append((11,size))
                    actions.append((9,size))
        if len(actions) == 0:
            if len(np.concatenate(list(origData['Positions'].values()))) == 0:
                return ((2,size), (9,size))
            else:
                return ((12,size),(12,size))
        if len(actions) == 1:
            actions.append((12,size))
        return actions

    def calculaterewards(self, termination) -> Any:
        penalty = self.rewardpenalty * (self.countInventory()**2)
        self.profit = self.cash - self.statelog[0][1]
        self.updatestatelog()
        deltaPNL = self.statelog[-1][2] - self.statelog[-2][2]
        deltaInv = self.statelog[-1][3]['INTC']*self.statelog[-1][-1]*(1- self.transaction_cost*np.sign(self.statelog[-1][3]['INTC'])) - self.statelog[-2][3]['INTC']*self.statelog[-2][-1]*(1-self.transaction_cost*np.sign(self.statelog[-1][3]['INTC']))
        return deltaPNL + deltaInv - penalty

from graphviz import Source

# Paste your DOT code here
full_dot_code = """
digraph TradingLogic {
    rankdir=TB;
    node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11];
    edge [fontname="Arial", fontsize=10];
    
    // Color scheme for clarity
    node [fillcolor="#E8F4F8"] // Light blue for checks
    
    start [label="get_action(data, ε=0.1)", fillcolor="#B8E6B8", shape=oval, fontsize=12, style="filled,bold"];
    
    // Breach check
    breach [label="Breach\nDetected?", fillcolor="#FFE5E5", shape=diamond];
    breach_action [label="Return:\nMO(4,1) if inv>0\nMO(7,1) if inv≤0", fillcolor="#FFB8B8"];
    
    // Lambda analysis
    lambda_check [label="Max Lambda\nCategory?", fillcolor="#FFF4E5", shape=diamond, width=1.8];
    
    // Lambda 4 branch (MO Ask hits ask)
    lambda4 [label="λ[4] Dominant\n(MO Ask)", fillcolor="#E5F0FF", style="filled,bold"];
    inv_neg [label="inv < -threshold?", shape=diamond, fillcolor="#E8F4F8"];
    mo_bid [label="Return:\n(12: MO_BID, 7: MO)", fillcolor="#D4F1D4"];
    
    inv_pos_good [label="inv > threshold\nOR\n(inv>0 & pa>PnL/inv)?", shape=diamond, fillcolor="#E8F4F8"];
    goodfill_actions4 [label="Good Fill Actions:\n• LO_BID_TOP(9) if na/qa>1\n• LO_ASK_TOP(2) if nb/qb<1", fillcolor="#D4F1D4"];
    
    inv_pos_bad [label="inv>0 & pa<PnL/inv?", shape=diamond, fillcolor="#E8F4F8"];
    badfill_actions4 [label="Bad Fill Actions:\n• LO_ASK_L1(3) if Ask_L1≠∅\n• LO_BID_TOP(9) if na/qa>1\n• LO_ASK_TOP(3) if nb/qb<1", fillcolor="#FFD4D4"];
    
    // Lambda 7 branch (MO Bid hits bid)
    lambda7 [label="λ[7] Dominant\n(MO Bid)", fillcolor="#E5F0FF", style="filled,bold"];
    inv_pos_7 [label="inv > threshold?", shape=diamond, fillcolor="#E8F4F8"];
    mo_ask [label="Return:\n(12: MO_ASK, 4: MO)", fillcolor="#D4F1D4"];
    
    inv_neg_good [label="inv < -threshold\nOR\n(inv<0 & pb<PnL/inv)?", shape=diamond, fillcolor="#E8F4F8"];
    goodfill_actions7 [label="Good Fill Actions:\n• LO_BID_TOP(9) if na/qa<1\n• LO_ASK_TOP(2) if nb/qb>1", fillcolor="#D4F1D4"];
    
    inv_neg_bad [label="inv<0 & pb>PnL/inv?", shape=diamond, fillcolor="#E8F4F8"];
    badfill_actions7 [label="Bad Fill Actions:\n• LO_BID_L1(8) if Bid_L1≠∅\n• LO_BID_TOP(9) if na/qa<1\n• LO_ASK_TOP(2) if nb/qb>1", fillcolor="#FFD4D4"];
    
    // Other lambda (skew-based)
    lambda_other [label="Other λ\n(Skew Strategy)", fillcolor="#E5F0FF", style="filled,bold"];
    skew_pos [label="skew > 0.5?", shape=diamond, fillcolor="#E8F4F8"];
    skew_neg [label="skew < -0.5?", shape=diamond, fillcolor="#E8F4F8"];
    
    skew_pos_actions [label="Positive Skew Actions:\n• LO_ASK_L2(1) if Ask_L2≠∅\n• LO_ASK_TOP(2)\n  [when (qa-nb)/mean < skew]\n• LO_BID_L2(11) if Bid_L2≠∅\n• LO_BID_TOP(9)\n  [when (na-qb)/mean < skew]", fillcolor="#F0E5FF"];
    
    skew_neg_actions [label="Negative Skew Actions:\n• LO_ASK_L2(1) if Ask_L2≠∅\n• LO_ASK_TOP(2)\n  [when (qa-nb)/mean > skew]\n• LO_BID_L2(11) if Bid_L2≠∅\n• LO_BID_TOP(9)\n  [when (na-qb)/mean > skew]", fillcolor="#F0E5FF"];
    
    // Final checks
    action_check [label="actions\nempty?", shape=diamond, fillcolor="#FFE5E5"];
    positions_check [label="positions\nempty?", shape=diamond, fillcolor="#E8F4F8"];
    default_both [label="Return:\n(2: LO_ASK_TOP,\n9: LO_BID_TOP)", fillcolor="#D4F1D4"];
    cancel_both [label="Return:\n(12: CANCEL,\n12: CANCEL)", fillcolor="#D4F1D4"];
    
    single_action [label="actions.len==1?", shape=diamond, fillcolor="#E8F4F8"];
    add_cancel [label="Add (12: CANCEL)", fillcolor="#D4F1D4"];
    return_actions [label="Return:\nactions", fillcolor="#B8E6B8", style="filled,bold", shape=oval];
    
    // Main flow
    start -> breach;
    breach -> breach_action [label="Yes", color=red, fontcolor=red];
    breach -> lambda_check [label="No", color=green4, fontcolor=green4];
    
    lambda_check -> lambda4 [label="λ[4]"];
    lambda_check -> lambda7 [label="λ[7]"];
    lambda_check -> lambda_other [label="Other"];
    
    // Lambda 4 flow
    lambda4 -> inv_neg;
    inv_neg -> mo_bid [label="Yes", color=red, fontcolor=red];
    inv_neg -> inv_pos_good [label="No"];
    inv_pos_good -> goodfill_actions4 [label="Yes", color=green4, fontcolor=green4];
    inv_pos_good -> inv_pos_bad [label="No"];
    inv_pos_bad -> badfill_actions4 [label="Yes", color=red, fontcolor=red];
    
    // Lambda 7 flow
    lambda7 -> inv_pos_7;
    inv_pos_7 -> mo_ask [label="Yes", color=red, fontcolor=red];
    inv_pos_7 -> inv_neg_good [label="No"];
    inv_neg_good -> goodfill_actions7 [label="Yes", color=green4, fontcolor=green4];
    inv_neg_good -> inv_neg_bad [label="No"];
    inv_neg_bad -> badfill_actions7 [label="Yes", color=red, fontcolor=red];
    
    // Skew flow
    lambda_other -> skew_pos;
    skew_pos -> skew_pos_actions [label="Yes"];
    skew_pos -> skew_neg [label="No"];
    skew_neg -> skew_neg_actions [label="Yes"];
    
    // Convergence to final check
    goodfill_actions4 -> action_check;
    badfill_actions4 -> action_check;
    inv_pos_bad -> action_check [label="No"];
    
    goodfill_actions7 -> action_check;
    badfill_actions7 -> action_check;
    inv_neg_bad -> action_check [label="No"];
    
    skew_pos_actions -> action_check;
    skew_neg_actions -> action_check;
    skew_neg -> action_check [label="No"];
    
    // Final checks
    action_check -> positions_check [label="Yes", color=red, fontcolor=red];
    action_check -> single_action [label="No", color=green4, fontcolor=green4];
    
    positions_check -> default_both [label="Yes"];
    positions_check -> cancel_both [label="No"];
    
    single_action -> add_cancel [label="Yes"];
    single_action -> return_actions [label="No"];
    add_cancel -> return_actions;
    
    default_both -> return_actions;
    cancel_both -> return_actions;
    mo_bid -> return_actions;
    mo_ask -> return_actions;
    breach_action -> return_actions;
    
    // Legend
    subgraph cluster_legend {
        label="Legend";
        fontsize=11;
        style=filled;
        fillcolor="#F9F9F9";
        
        leg1 [label="Decision Point", shape=diamond, fillcolor="#E8F4F8"];
        leg2 [label="Action/Return", shape=box, fillcolor="#D4F1D4", style="rounded,filled"];
        leg3 [label="Strategy Branch", shape=box, fillcolor="#E5F0FF", style="filled,bold"];
        leg4 [label="Bad Fill (Risk)", shape=box, fillcolor="#FFD4D4", style="rounded,filled"];
        
        leg1 -> leg2 -> leg3 -> leg4 [style=invis];
    }
    
    // Notes
    note [label="Key Variables:\n• inv: inventory position\n• pa, pb: best ask/bid prices\n• na, nb: net ask/bid counts\n• qa, qb: quote ask/bid\n• skew: (na-nb)/mean(qa,qb)\n• PnL: (cash-init_cash)", shape=note, fillcolor="#FFFACD", fontsize=9];
}
"""
dot_code = """
digraph TradingStrategy {
    rankdir=TB;
    node [shape=box, style="rounded,filled", fontname="Arial Bold", fontsize=13];
    edge [fontname="Arial", fontsize=11, penwidth=2];
    
    // Start
    start [label="Trading Decision", fillcolor="#4A90E2", fontcolor=white, shape=oval, width=2];
    
    // Level 1: Emergency check
    breach [label="Emergency\nBreach?", fillcolor="#E74C3C", fontcolor=white, shape=diamond, width=1.8];
    emergency [label="CLOSE OUT\nImmediately", fillcolor="#C0392B", fontcolor=white, penwidth=3];
    
    // Level 2: Market direction
    market_flow [label="Which Side is\nMarket Hitting?", fillcolor="#9B59B6", fontcolor=white, shape=diamond, width=2.2];
    
    ask_side [label="📈 ASK SIDE\n(Buying Pressure)", fillcolor="#3498DB", fontcolor=white, width=2];
    bid_side [label="📉 BID SIDE\n(Selling Pressure)", fillcolor="#E67E22", fontcolor=white, width=2];
    balanced [label="⚖️ BALANCED\n(Use Skew)", fillcolor="#16A085", fontcolor=white, width=2];
    
    // Level 3: Position evaluation
    ask_inv [label="Too Much\nInventory?", fillcolor="#5DADE2", shape=diamond];
    bid_inv [label="Too Much\nInventory?", fillcolor="#F39C12", shape=diamond];
    skew_dir [label="Skew\nDirection?", fillcolor="#48C9B0", shape=diamond];
    
    // Level 4: Actions
    ask_good [label="✅ GOOD FILL\nPlace passive orders\nto unwind position", fillcolor="#27AE60", fontcolor=white, penwidth=2];
    
    ask_bad [label="❌ BAD FILL\nAggressive orders\nto exit quickly", fillcolor="#E74C3C", fontcolor=white, penwidth=2];
    
    bid_good [label="✅ GOOD FILL\nPlace passive orders\nto unwind position", fillcolor="#27AE60", fontcolor=white, penwidth=2];
    
    bid_bad [label="❌ BAD FILL\nAggressive orders\nto exit quickly", fillcolor="#E74C3C", fontcolor=white, penwidth=2];
    
    skew_actions [label="📊 SKEW STRATEGY\nPlace orders on both sides\nbased on imbalance", fillcolor="#1ABC9C", fontcolor=white, penwidth=2];
    
    done [label="Execute Trade", fillcolor="#2ECC71", fontcolor=white, shape=oval, width=2, penwidth=3];
    
    // Flow
    start -> breach;
    breach -> emergency [label="YES", color="#E74C3C", fontcolor="#E74C3C", penwidth=3];
    breach -> market_flow [label="NO", color="#27AE60", fontcolor="#27AE60"];
    
    market_flow -> ask_side [label="Asks"];
    market_flow -> bid_side [label="Bids"];
    market_flow -> balanced [label="Neither"];
    
    ask_side -> ask_inv;
    ask_inv -> ask_good [label="NO\n(profitable)", color="#27AE60", fontcolor="#27AE60"];
    ask_inv -> ask_bad [label="YES\n(losing)", color="#E74C3C", fontcolor="#E74C3C"];
    
    bid_side -> bid_inv;
    bid_inv -> bid_good [label="NO\n(profitable)", color="#27AE60", fontcolor="#27AE60"];
    bid_inv -> bid_bad [label="YES\n(losing)", color="#E74C3C", fontcolor="#E74C3C"];
    
    balanced -> skew_dir;
    skew_dir -> skew_actions [label="Either"];
    
    emergency -> done;
    ask_good -> done;
    ask_bad -> done;
    bid_good -> done;
    bid_bad -> done;
    skew_actions -> done;
    
    // Legend
    subgraph cluster_legend {
        label="CORE CONCEPT";
        fontsize=14;
        fontname="Arial Bold";
        style="filled,bold";
        fillcolor="#ECF0F1";
        penwidth=2;
        
        concept [label="Good Fill = Making money, take time to exit\nBad Fill = Losing money, exit fast", 
                 shape=note, fillcolor="#FFFACD", fontsize=12, style="filled,bold"];
    }
}
"""
# Render to file
src = Source(dot_code)
src.render('trading_logic_graph', format='png', view=True)

# This will:
# 1. Create trading_logic_graph.png
# 2. Automatically open it in your default image viewer
print("Graph rendered successfully!")