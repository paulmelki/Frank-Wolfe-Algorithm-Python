# Frank-Wolfe Algorithm in Python
## Le Anh DUNG and Paul MELKI (Toulouse School of Economics)
---
Implementation of the Frank-Wolfe optimization algorithm in Python with an application for solving the LASSO problem.

Some useful resources about the Frank-Wolfe algorithm can be found here: 
- http://fa.bianp.net/blog/2018/notes-on-the-frank-wolfe-algorithm-part-i/
- https://people.csail.mit.edu/stefje/fall15/notes_lecture14.pdf
- https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/frank-wolfe.pdf

This implementation is divided over two files:
- `frank_wolfe.py`: in this file we define the functions required for the implementation of the Frank-Wolfe algorithm, as well as the function `frankWolfeLASSO` which solves a LASSO optimization problem using the algorithm.
- `implementation_LASSO.py`: in this file we solve different LASSO problems using our implementation, which allows to look at the *sensitivity* of the algorithm towards different changes in the problem's definition: for example, changes in the number of observations (n), or the number of parameters (p).

This code has been developed as part of the final project for the course *Optimization for Big Data* in M1 Statistics and Econometrics at Toulouse School of Economics. 
