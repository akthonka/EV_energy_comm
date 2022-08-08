# Electric Vehicles in Energy Communities: Investigating the Distribution Grid Hosting Capacity
BSc SSE Thesis of Daniil Aktanka

### Creating a case network
1. Randomly select load profile for a single day
2. Limit the profile to night window
3. Create a generator column (sgen)
4. Select a time window. Assign power value to the sgen column
5. Repeat steps 1-4 for desired number of loads/sgens in the network

 ___

### Case variables
- Number of sgens can vary from 1 to max
- Charging time window (of 1hr?) can vary between _20:00_ and _6:00_
- Number of coordinating house owners can vary from 0 to all

        - Parties don't charge EVs at the same time; assume "one after another" approach
        - Neigboring parties can reduce local load. Distant partners reduce overall load


### Evaluation criteria and metrics
- Max line load (inc. overload)
- Visualisation of network line load


### Scenarios
- Uncoordinated charging
- (Semi-)coordinated charging
- Critical scenario: max number of EVs, all charging at the same time

___

### Energy community effectiveness evaluation
Consider _n_ cases (i.e. variations of network loads) per scenario (i.e. coordinated / uncoordinated charging)

FOR _x_ IN range(_n_) :
        
        1. Create case network from random historical load profiles
        2. Run timeseries on time window
        3. Calculate max line load(s), save output to df

Calculate probability of line overload per scenario.