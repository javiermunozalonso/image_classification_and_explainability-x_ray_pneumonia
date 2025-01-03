def add_item(list):
    list.append('A')
    return list

def reset_list(list):
    list = []
    return list

my_list = ['a', 'b', 'c']
add_item(my_list)
add_item(my_list)
add_item(my_list)
add_item(my_list)
add_item(my_list)
print(my_list)
my_list = reset_list(my_list)
print(my_list)