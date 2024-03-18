from tqdm import tqdm
import collections


def load_ratings(file):
    inters = []
    with open(file, 'r') as fp:
        count = 0
        for line in tqdm(fp, desc='Load ratings'):
            count += 1
            if count == 1:
                continue
            # print(line)
            user, item, timestamp = line.strip().split()[:3]
            inters.append((user, item, int(timestamp)))
            # print(inters)
    return inters


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, timestamp = inter
        user2inters[user].append((user, item, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[2])
        for inter in user_inters:
            new_inters.append(inter)
    return user2inters, new_inters


def write(user2inters):
    lengths = [256, 512, 1000, 2000, 5000, 10000]
    for length in lengths:
        count = 0
        file = './tracks' + str(length) + '.txt'
        with open(file, 'w') as f:
            for user in user2inters:
                if count > 25000:
                    break
                count += 1 
                for inter in user2inters[user][-length:]:
                        user = inter[0]
                        item = inter[1]
                        f.write(str(user))
                        f.write(' ')
                        f.write(str(item))
                        f.write('\n')


file = ""
inters = load_ratings(file)
user2inters, new_inters = make_inters_in_order(inters)
write(user2inters)