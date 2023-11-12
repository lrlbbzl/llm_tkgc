import json
import argparse
import os

def generate_data(args):
    train_data_path = os.path.join(args.data_path, 'train.txt')
    valid_data_path = os.path.join(args.data_path, 'valid.txt')
    test_data_path = os.path.join(args.data_path, 'test.txt')
    
    ent_path, rel_path = os.path.join(args.data_path, 'entity2id.txt'), os.path.join(args.data_path, 'relation2id.txt')
    
    single_ts_ent = dict() # record entities showing up in each timestamp
    origin2order = dict() # record entities' id transformation in each timestamp

    id2ent, id2rel = dict(), dict()
    for line in open(ent_path, 'r').readlines():
        name, id = line.strip().split('\t')
        id2ent.update({int(id) : name})

    for line in open(rel_path, 'r').readlines():
        name, id = line.strip().split('\t')
        id2rel.update({int(id) : name})
    
    ent_num, rel_num = len(id2ent), len(id2rel)
    additional_triples = []

    for line in open(train_data_path, 'r').readlines():
        h, r, t, ts = line.strip().split('\t')
        h, r, t, ts = int(h), int(r), int(t), int(ts)


        if ts not in single_ts_ent:
            single_ts_ent.update({ts : set()})
        if ts not in origin2order:
            origin2order.update({ts : dict()})

        if h not in single_ts_ent[ts]:
            single_ts_ent[ts].add(h)
            # if len(single_ts_ent.keys()) == 1:
            #     origin2order[ts].update({h : h})
            if min(single_ts_ent.keys()) < ts:
                # start finding connected entity
                for k, v in single_ts_ent.items():
                    if k != ts - 1:
                        continue
                    delta_t = ts - k

                    if h in v:
                        # add temporal entity
                        if h not in origin2order[ts]:
                            # have not get specific new number
                            new_id = ent_num
                            ent_num += 1
                            id2ent.update({new_id : id2ent[h]})
                            origin2order[ts].update({h : new_id})
                        else:
                            new_id = origin2order[ts][h]
                        pre_ts_ent = origin2order[k][h] if h in origin2order[k] else h
                        # add relation
                        rel_id = rel_num - 1 + delta_t
                        if rel_id not in id2rel:
                            id2rel.update({rel_id : "t_{}".format(delta_t)})
                        additional_triples.append((pre_ts_ent, rel_id, new_id))
        
        if t not in single_ts_ent[ts]:
            single_ts_ent[ts].add(t)
            # if len(single_ts_ent.keys()) == 1:
            #     origin2order[ts].update({t : t})
            if min(single_ts_ent.keys()) < ts:
                # start finding connected entity
                for k, v in single_ts_ent.items():
                    if k >= ts:
                        continue
                    delta_t = ts - k

                    if t in v:
                        # add temporal entity
                        if t not in origin2order[ts]:
                            new_id = ent_num
                            ent_num += 1
                            id2ent.update({new_id : id2ent[t]})
                            origin2order[ts].update({t : new_id})
                        else:
                            new_id = origin2order[ts][t]

                        pre_ts_ent = origin2order[k][h] if h in origin2order[k] else h
                        # add relation
                        rel_id = rel_num - 1 + delta_t
                        if rel_id not in id2rel:
                            id2rel.update({rel_id : "t_{}".format(delta_t)})
                        additional_triples.append((pre_ts_ent, rel_id, new_id))

    print(len(additional_triples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM for TKGC')
    parser.add_argument('--data_path', type=str, default='./data/ICEWS14', help='Prepared data path.')

    args = parser.parse_args()
    generate_data(args)