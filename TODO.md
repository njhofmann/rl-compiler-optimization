1. integrate termination criteria into env via method overload?
    1. patience - no improvement after n steps terminate, else reset to n
    1. patience but don't reset after improvement, only decrement?
    1. run for n steps everytime
    1. override agent functions?
    1. convergence? 0 reward negative or positive
   
1. track eval criteria with call back
    1. cum reward per episode    
    1. q value for states over time
    1. use raw policy prediction scores for q-value tracking
   
1. track real time as well
1. assign specific benchmarks to envs
1. track worthwhile papers
1. how is different from autphase paper   