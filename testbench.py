from MailFilter import MailFilter
from time import perf_counter
import time

tic = perf_counter()

mailFilter = MailFilter()
mailFilter.conn_init()
#mailFilter.train()
#mailFilter.test()

while True:
    mailFilter.apply_to_account()
    time.sleep(3)

toc = perf_counter()

print(toc - tic)