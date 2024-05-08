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
            user, item, date, hour_min, time_ms, is_click = line.strip().split(',')[:6]
            if is_click:
                inters.append((user, item, int(time_ms)))
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
    lengths = [1000, 2000, 5000, 10000]
    for length in lengths:
        file = './KuaiRand' + str(length) + '.txt'
        with open(file, 'w') as f:
            for user in user2inters:
                count = 0
                for inter in user2inters[user]:
                    count += 1
                    if count < length:
                        user = inter[0]
                        item = inter[1]
                        f.write(str(user))
                        f.write(' ')
                        f.write(str(item))
                        f.write('\n')


file = "/data/irlab_share/KuaiRand/log_standard_4_08_to_4_21_27k_part1.csv"
inters = load_ratings(file)
user2inters, new_inters = make_inters_in_order(inters)
write(user2inters)
