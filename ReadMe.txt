Some explanation about my work on this assignment:

1. When running my code, you could encounter a value warning about the invalid value in the calculation. It is not unexpected because basinhopping function tries different parameter values to find global minimums, so there could be a condition that the function is trying a parameter value which leads to a negative value in log function or square root function. However, this will not influence our results since it is just one try of root finding. I use "warnings.filterwarnings("ignore")" to ignore the warnings, if you want to check whether warning happens, you can remove this code.

2. The reason why I use "np.random.seed(3)" to set a seed for optimization is, basinhopping function randomly searches for the next possible solution. Setting seed makes sure we get the same result for each run. And I use seed "3" because it greatly reduces the error when tenor=1 day while other seeds could just reduce the error to around 0.12.

3. The reason why I set iteration number=1000 only for tenor=1 day while others are 100 is, it is hard to reduce the error for tenor=1 day below 1e-4 unless we increase iteration number for it. We fix other iteration numbers on 100 to improve efficiency.

4. "Result1_DataTable.csv" gathers the results including optimized K_ATM, K_RR_Call, K_RR_Put, K_BF_Call, K_BF_Put, corresponding implied volatilities, and calculated K_Call with delta=0.1 and K_Put with delta=-0.1 as the whole answers to question 1 and one part of the answer to question 2.

"Result2_VolatilityCurve.png" gives the relationships between K and sigma for different tenors as another part of the answer to question 2.

"MyCodeForComputingAssignment.py" is the optimization code for this assignment.

5. The running time is about 6 minutes on my computer mainly due to the high number of iterations for t=1 day to get enough precision. Time may vary depending on your computer's capacity and working environment.

