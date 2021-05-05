1. integrate termination criteria into env via method overload?
    1. convergence? 0 reward negative or positive
   
1. track eval criteria with call back
    1. cum reward per episode    
    1. q value for states over time
    1. use raw policy prediction scores for q-value tracking
   
1. to test
   1. different envs
   1. different termination criteria
      1. 25, 50, 100, 1000 iterations
      1. 25 50 100 patience
      1. 1, 2, 5 real life secs?
      1. pre-emptive termination?
   1. k prev actions - 1, 5, 10, 100
      1. depending on termination criteria
   
1. track worthwhile papers
   1. explain how we differ
1. k actions at a time
1. hyperparam search   
1. additional datasets
   1. specific training data?
1. implement baseline methods
   1. random search
   1. greedy search
1. additional termination criteria
   1. simulated annealing
   1. give optional 