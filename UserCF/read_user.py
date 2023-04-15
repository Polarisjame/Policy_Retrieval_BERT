def read_user_data(file_root):
    with open(file_root, 'r') as f:
        user_item_dict = dict()
        titles = ['职业', '省份', '感兴趣的领域']
        user_attributes_dict = {title: dict() for title in titles}
        for i in f:
            User_inf = [int(a) for a in i.split()]
            id = User_inf[0]
            judge = 0
            number = 0
            last_number = 0
            index = 0
            items = []
            for ind in range(1, len(User_inf)):
                if not judge:
                    number = User_inf[ind]
                    last_number = number
                    judge = 1 - judge
                else:
                    if number > 0:
                        if index == 3:
                            items.append(User_inf[ind])
                            number -= 1
                        else:
                            title = titles[index]
                            if id not in user_attributes_dict[title]:
                                user_attributes_dict[title][id] = {User_inf[ind]}
                            else:
                                user_attributes_dict[title][id].add(User_inf[ind])
                            number -= 1
                        if number == 0:
                            index += 1
                            judge = 1 - judge
            user_item_dict[id] = set(items)
        return user_attributes_dict, user_item_dict
