 
# Домашнее задание №4 по курсу "Высокопроизводительные вычисления" Соколов Александр
 


# Программа расчета количества возможных путей.
путь к программе /srs/matmul_blas.c

Для расчета всех возможных путей использована формула

R = E + A + A^2 +A^3 .....+A^n
где n -размер матрицы, А - матрица связности графа, E единичная матрица.

количеставо элекментов вычисляется как сумма элементоа матрицы R

для возведения в степень матрицы А используются предварительно рассчитанные матрицы А^n , где n - степень числа 2

расчет этих матриц осуществляется последовательным взведением в квадрат исходной матрици связности.


Для операций умножения матриц используется библиотера BLAS

Операции сложения и инициализации матриц оптимизированы для использования openmp

компиляция программы  gcc matmul_blas.c -o 1 -lcblas -fopenmp

