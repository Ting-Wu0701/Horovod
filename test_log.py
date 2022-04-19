import time

N = 2000000

for idx in range(N):
    tt = time.localtime(time.time())
    # print("{}".format(idx))
    print("{},{}".format(idx, tt))

    # if idx % 5 == 0:
    	# time.sleep(1)