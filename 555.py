#
# a = {'a':1}
# b = str(a)
# print(type(b))
# print(b)

# string = 'IIIIBIIIBIIBOOOOOO'
# print(string.count('B'))
# print(string.index('B'))
# tt = ''.join(reversed(string))
# index1 = string.index('B')
# index2 = len(tt)-tt.index('B')-1
# print(string[index2])
# print()
# print(string.find('B'))

a = []
print('%.5f,sdfsa %.2f'% (0.5555555, 1.255566))

import torch

uv = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
ab = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
cc = torch.einsum('ik,kj->ij', uv, ab)
print(cc)