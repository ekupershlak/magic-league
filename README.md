# magic-league
Super-Swiss pairing algorithm for casual Magic: The Gathering leagues

At my workplace, we have a recurring Magic: the Gathering sealed league. A problem we faced was that people needed flexibility to play varying numbers of matches. We also wanted to maintain the benefits of Swiss pairings.

This software uses the LKH solver to solve the Traveling Salesman optimization problem of assigning matches that respects requested numbers of matches while minimizing pairing differences in win-percentage among the pairings. It can also post pairings directly to a Google sheet that is specifically formatted.
