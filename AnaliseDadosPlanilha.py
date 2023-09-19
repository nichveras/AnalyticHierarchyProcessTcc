# coding=UTF-8

import xlrd
import matplotlib.pyplot as plt

arq=xlrd.open_workbook("dados.xlrx")
plan=arq.sheet_by_name("dados")

x=plan.col_values(0)
y=plan.col_values(6)
y=plan.col_values(7)
y=plan.col_values(8)
y=plan.col_values(9)
y=plan.col_values(10)
y=plan.col_values(11)
y=plan.col_values(12)
y=plan.col_values(13)
y=plan.col_values(14)
y=plan.col_values(15)
y=plan.col_values(16)
y=plan.col_values(17)
y=plan.col_values(18)
y=plan.col_values(19)
y=plan.col_values(20)
y=plan.col_values(21)
y=plan.col_values(22)

plt.plot(x,y,color='red', linewidth=2.0)
plt.show()




