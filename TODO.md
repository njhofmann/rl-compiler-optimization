1. integrate termination criteria into env via method overload?
    1. convergence? 0 reward negative or positive
   
1. track eval criteria with call back
    1. cum reward per episode    
    1. q value for states over time
    1. use raw policy prediction scores for q-value tracking
   
1. environment shaping
   1. termination criteria    
      1. interacts with eval how?
   
1. hyperparameter optimization with bayesian optimization
   1. how to deal with discretes?
   1. \# of parameters vs number of trials
   
1. random seeds
   1. how many? 
   1. vs trials?
   
1. reward shaping
   1. reward function as is 
      1. positive improvement --> positive reward
      1. negative improvement --> negative reward
      1. no improvement --> no reward
   1. observe many actions have no improvement, discourage such actions via neutral penalty?
      1. ie negative reward if no improvement, but smaller than negative improvement rewards, don't want to discourage
         exploration
      1. want to encourage more positive improvements faster
   
1. in paper
   1. no concept of terminal state for episode aside from errors
      1. we apply our own "termination criteria" to encourage desired behavior
         1. desired behavior == "agent applies actions leading to positive improvement"
   
1. to test
   1. different envs
   1. different termination criteria
      1. 25, 50, 100, 1000 iterations
      1. 25 50 100 patience
      1. 1, 2, 5 real life secs?
      1. pre-emptive termination?
   1. k prev actions - 1, 5, 10, 100
      1. depending on termination criteria
   1. different tasks
   1. 5 trials?
   
1. track worthwhile papers
   1. explain how we differ
1. k actions at a time
1. hyperparam search with ray tune  
1. additional datasets
   1. specific training data?
1. implement baseline methods
   1. random search
   1. greedy search
1. additional termination criteria
   1. simulated annealing
   1. give optional 