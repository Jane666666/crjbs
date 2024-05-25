import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']=['SimSun']
# with open("output.txt", "r") as f:
#     data = f.readlines()

# data = [float(x.strip()) for x in data]
x  = [1	,2	,3	,4	,5	,6	,7	,8	,9	,10	,11	,12	,13	,14	,15	,16	,17	,18	,19	,20	,21	,22	,23	,24	,25	,26	,27	,28	,29	,30	,31	,32	,33	,34	,35	,36	,37	,38	,39	,40	,41	,42	,43	,44	,45	,46	,47	,48	,49	,50	,51	,52	,53	,54	,55	,56	,57	,58	,59	,60	,61	,62	,63	,64	,65	,66	,67	,68	,69	,70	,71	,72	,73	,74	,75	,76	,77	,78	,79	,80	,81	,82	,83	,84	,85	,86	,87	,88	,89	,90	,91	,92	,93	,94	,95	,96	,97	,98	,99	,100]

y  = [1	,2	,3	,4	,5	,6	,7	,8	,9	,10	,11	,12	,13	,14	,15	,16	,17	,18	,19	,20	,21	,22	,23	,24	,25	,26	,27	,28	,29	,30	,31	,32	,33	,34	,35	,36	,37	,38	,39	,40	,41	,42	,43	,44	,45	,46	,47	,48	,49	,50	,51	,52	,53	,54	,55	,56	,57	,58	,59	,60	,61	,62	,63	,64	,65	,66	,67	,68	,69	,70	,71	,72	,73	,74	,75	,76	,77	,78	,79	,80	,81	,82	,83	,84	,85	,86	,87	,88	,89	,90	,91	,92	,93	,94	,95	,96	,97	,98	,99	,100]
plt.figure(figsize=(10, 10), dpi=100)

# plt.scatter(years, p,c='red',s=10, alpha=0.8, label='prior')
# plt.scatter(years, h,c='darkgreen',s=40, alpha=0.8, label='agent_her2')
# plt.scatter(years, dh, c ='m',s=40, alpha=0.8, label='agent_DH')
# plt.scatter(years, d, c ='darkorange',s=40, alpha=0.8, label='agent_D')
# plt.scatter(years, her, c='chocolate' ,s=40, alpha=0.8, label='agent_her2')
# plt.scatter(years, mpo ,s=40, alpha=0.8, label='agent_mpo')
# plt.scatter(years, h, c='darkgreen' ,s=40, alpha=0.8, label='agent_her2')
plt.scatter(y, x, c='darkviolet' ,s=40, alpha=0.8, label='agent_mpo')
plt.xlabel('生成序列',size = 36)
plt.ylabel('her特异性得分',size = 36)
plt.legend(fontsize = 26)
plt.xticks(fontsize = 30) 
plt.yticks(fontsize=30)

plt.show()
