import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif']=['SimSun']
# with open("data.txt", "r") as f:
#     data = f.readlines()

# data = [float(x.strip()) for x in data]
f = 'output.txt'
a = np.loadtxt(f, skiprows=2,usecols=(1,) )
# a = np.loadtxt(f)
x = a[5:105]
y = [1	,2	,3	,4	,5	,6	,7	,8	,9	,10	,11	,12	,13	,14	,15	,16	,17	,18	,19	,20	,21	,22	,23	,24	,25	,26	,27	,28	,29	,30	,31	,32	,33	,34	,35	,36	,37	,38	,39	,40	,41	,42	,43	,44	,45	,46	,47	,48	,49	,50	,51	,52	,53	,54	,55	,56	,57	,58	,59	,60	,61	,62	,63	,64	,65	,66	,67	,68	,69	,70	,71	,72	,73	,74	,75	,76	,77	,78	,79	,80	,81	,82	,83	,84	,85	,86	,87	,88	,89	,90	,91	,92	,93	,94	,95	,96	,97	,98	,99	,100]
plt.figure(figsize=(10, 10), dpi=100)
plt.scatter(y, x, c='darkviolet' ,s=40, alpha=0.8)
plt.xlabel('seq',size = 36)
plt.ylabel('score',size = 36)
# plt.legend(fontsize = 26)
plt.xticks(fontsize = 30) 
plt.yticks(fontsize=30)

# plt.show()

plt.savefig('./test.png')
plt.close()