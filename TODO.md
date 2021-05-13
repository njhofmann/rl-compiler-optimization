1. track eval criteria with call back
    1. cum reward per episode    
    1. q value for states over time
    1. use raw policy prediction scores for q-value tracking

1. experiments to run
   1. 20 trials of bayesian hyperparam opt for each model on same dataset
      1. use best hyperparams on future experiments
   1. different termination criteria for three diff datasets for each model
   1. env shaping w/ best performing termination criteria
   1. 9 trials for each experiment , 3 random seeds * 3 trials
   
1. in paper
   1. explain how we differ from previous RL works
   1. explain motivation for using deep RL   
   1. no concept of terminal state for episode aside from errors
      1. we apply our own "termination criteria" to encourage desired behavior
         1. desired behavior == "agent applies actions leading to positive improvement"
   1. env alterations to improve learning
      1. negative reward to for no improvement
         1. smaller than negative rewards for negative improvements
         1. just want to discourage these states more
      1. k prev actions to give better idea what occurred in the past
   
1. implement baseline methods
   1. random search
   1. greedy search
1. additional termination criteria
   1. simulated annealing
   1. give optional 