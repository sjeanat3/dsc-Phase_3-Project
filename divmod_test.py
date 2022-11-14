number = 17
cols = 4
my_list = []

for num in range(number):
    my_list.append(divmod(num, cols))
rows, rem = divmod(number, cols)
if rem:
    rows += 1
shape = (rows, cols)

suffix = ""
if suffix:
    print('yes')
else:
    print('no')
