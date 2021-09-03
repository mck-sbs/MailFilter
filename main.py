from MailFilter import MailFilter
from time import perf_counter

tic = perf_counter()

mailFilter = MailFilter()
mailFilter.conn_init()
#mailFilter.train()
#mailFilter.test()

toc = perf_counter()

print(toc - tic)