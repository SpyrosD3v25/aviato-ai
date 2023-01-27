import findspark
findspark.init()
from pyspark import SparkContext

import networkx as nx
from datetime import datetime

#The user defined parameters
money_cut_off = 10000
commission_cut_off = 0.8
group_by_time = 1 #days

sc = SparkContext(appName = "graph")
lines = sc.textFile('./small_transactions.csv')
header = lines.first()
lines = lines.filter(lambda x: x != header)
lines = lines.filter(lambda x: x.split('|')[1] != '2006-02-29')


def get_amount_per_node_transfer(lines):
    filtered_by_amount_per_node = {key: amount for key, amount in lines.map(lambda x: (
                                    (datetime.strptime(x.split('|')[1],'%Y-%m-%d'
                                    ).toordinal() // group_by_time, x.split('|')[3]),
                                    float(x.split('|')[2]))).reduceByKey(lambda x,y: 
                                    x+y).filter(lambda x: x[1]>money_cut_off).collect()}
    
    return filtered_by_amount_per_node

filtered_by_amount_per_node = get_amount_per_node_transfer(lines)


def get_graph_edges(lines, filtered_by_amount_per_node):
    def filter_transactions(row):
        if (datetime.strptime(row.split('|')[1],'%Y-%m-%d'
            ).toordinal() // group_by_time, row.split('|')[3]) in filtered_by_amount_per_node.keys():
            return (datetime.strptime(row.split('|')[1],'%Y-%m-%d').toordinal() // group_by_time, 
                    row.split('|')[0], [[row.split('|')[3], row.split('|')[4], float(row.split('|')[2])]])
        else:
            return 0
    
    graph_edge_constructor = {key: data for key, data in lines.map(filter_transactions
                            ).filter(lambda x: x != 0).map(lambda x: (x[0], x[2])
                            ).reduceByKey(lambda x,y: x+y).collect()}
    
    filtered_transactions = {key: data for key, data in lines.map(filter_transactions
                            ).filter(lambda x: x != 0).map(lambda x: (x[:2], x[2])
                            ).collect()}
    
    return graph_edge_constructor ,filtered_transactions                           

graph_edge_constructor, filtered_transactions = get_graph_edges(lines, filtered_by_amount_per_node)


def create_graph(edges):
    G=nx.MultiDiGraph()
    G.add_weighted_edges_from(edges)
    return G


def get_depth(G, node):
    return max(nx.single_source_shortest_path_length(G, node).values())


def breadth_first_search(G, node):
    return list(nx.shortest_path(G, source=node))


def get_tansaction_traffic_per_node(G, time, node, transaction_volume):
    if (time, node) not in transaction_volume:
            transaction_volume[(time, node)] = (G.in_degree(node)
                                                    + G.out_degree(node))
    else:
        transaction_volume[(time, node)] += (G.in_degree(node)
                                                + G.out_degree(node))
    return transaction_volume


def get_normalized_edge_weights(G, time, connected_nodes, transaction_volume, filtered_by_amount_per_node):
    normalized_transactions = []
    for node in connected_nodes:
        neighbors = G.neighbors(node)
        transaction_volume = get_tansaction_traffic_per_node(G, time, node, 
                                                             transaction_volume)
        betweenness_dict[(time, node)] = betweenness_nodes[node]
        closeness_dict[(time, node)] = closeness_nodes[node]
        
        for neighbor in neighbors:
            #loop over the possible multiple edges connecting two nodes
            for k in range(len(G.get_edge_data(node, neighbor))):
                edge_amount = G.get_edge_data(node, neighbor)[k]['weight']
                initial_sent_amount = filtered_by_amount_per_node[tuple([time, connected_nodes[0]])]

                normalized_transactions.append([time, node, neighbor, edge_amount,
                                                edge_amount/initial_sent_amount])
    
    return (normalized_transactions, transaction_volume)


all_normalized_transactions, transaction_volume, betweenness_dict, closeness_dict = [], {}, {}, {}
###main function
for date, all_transactions_per_time in zip(graph_edge_constructor.keys(), graph_edge_constructor.values()):
    #Creates different graphs for different days
    G = create_graph([trans for trans in all_transactions_per_time])
    max_subgraph_size, visited_subgraphs = 0, []
    
    for transaction in all_transactions_per_time:
        # The depth of the transaction chain starting from the inital sender
        depth = get_depth(G, transaction[0])
        
        if depth > 1:
            #get connected nodes in bfs order
            connected_nodes = breadth_first_search(G, transaction[0])
            
            if max_subgraph_size  < len(connected_nodes) or not set(connected_nodes) <= set(visited_subgraphs):
                max_subgraph_size = len(connected_nodes)
                visited_subgraphs += connected_nodes
                # I looked at the closeness and betweenness for only 
                # transaction groups with multiple edges.
                # This saved significant amount of computational time 
                # as graph size decreases once the 
                # transactions with single edges are filtered.
                sub_G = G.subgraph(connected_nodes)
                betweenness_nodes = nx.betweenness_centrality(sub_G, normalized=True)
                closeness_nodes = nx.closeness_centrality(sub_G)
                
                # Edge weights are normalized by the money initially sent from the sender.
                # Also in and out degree weights are calculated for each node.
                normalized_transactions, transaction_volume = get_normalized_edge_weights(G, 
                                date, connected_nodes, transaction_volume, filtered_by_amount_per_node)
                
                if normalized_transactions not in all_normalized_transactions:
                    all_normalized_transactions.append(normalized_transactions)


def get_receivers(sender_id, all_day_transactions):
    receivers = []
    for transaction in all_day_transactions:
        if transaction[1] != sender_id:
            receivers.append(transaction[2])
    return receivers


def get_multiple_receivers(receivers):
    seen, multiple_receiver = [], set()
    for r in receivers:
        if r not in seen:
            seen.append(r)
        else:
            multiple_receiver.add(r)
    return multiple_receiver

def fix_multiple_receiver_amount(all_day_transactions, multiple_receivers):
    for r in multiple_receivers:
        r_sum, edge_sum = 0, 0
        
        for transaction in all_day_transactions:
            if transaction[2] == r:
                r_sum += transaction[4]
                edge_sum += transaction[3]
        
        if r_sum >= commission_cut_off and r_sum <= 1.:
            all_day_transactions.append([all_day_transactions[0][0], 'sender', r, edge_sum, r_sum])
    return all_day_transactions


def get_suspicious_transactions(all_normalized_transactions, commission_cut_off):
    suspicious_transactions = []
    for all_day_transactions in all_normalized_transactions:
        sender_id = all_day_transactions[0][1]
        receivers = get_receivers(sender_id, all_day_transactions)
        multiple_receivers = get_multiple_receivers(receivers)
        if multiple_receivers:
            all_day_transactions = fix_multiple_receiver_amount(all_day_transactions, 
                                                                multiple_receivers)
                
        for transaction in all_day_transactions:
            if sender_id != transaction[1] and transaction[4] >= commission_cut_off and transaction[4] <= 1.:
                suspicious_transactions.append(all_day_transactions)     
    return suspicious_transactions


#fix multiple edge receiver problem        
suspicious_transactions = get_suspicious_transactions(all_normalized_transactions, commission_cut_off)


def get_suspicious_traffic(transaction_volume):
    max_volume = max(transaction_volume.values())
    suspicious_traffic = {k: v / max_volume for k, v in transaction_volume.items()}
    return suspicious_traffic

#normalize the transaction traffic parameter
suspicious_traffic = get_suspicious_traffic(transaction_volume)


def get_score(suspicious_traffic, betweenness_dict, closeness_dict):
    score_day_node, score_dict = {}, {}
    a, b, c = 0.5, 0.4, 0.1
    for key in suspicious_traffic.keys():
        score_day_node[key] = a*suspicious_traffic[key] + b*betweenness_dict[key] + c*closeness_dict[key]
    
    for key in score_day_node.keys():
        if key[1] not in score_dict:
            score_dict[key[1]] = score_day_node[key]
        else:
            score_dict[key[1]] += score_day_node[key]
            
    sorted_score_nodes = sorted(score_dict.items(), key=lambda kv: -kv[1])
    sorted_score_nodes = [node[0] for node in sorted_score_nodes]
    return sorted_score_nodes, score_dict

score_nodes, score_dict = get_score(suspicious_traffic, betweenness_dict, closeness_dict)


def sort_suspicious_transactions(suspicious_transactions, score):
    suspicious_transactions_parameter = []
    for transaction_chain in suspicious_transactions:
        score_sum = 0
        for transaction in transaction_chain:
            if transaction[1] != 'sender':
                score_sum += score[transaction[1]] + score[transaction[2]]
        avg_score = score_sum/(2*(len(transaction_chain)))
        if (score_sum, transaction_chain) not in suspicious_transactions_parameter:
            suspicious_transactions_parameter.append((avg_score, transaction_chain))
    
    suspicious_transactions_parameter = sorted(suspicious_transactions_parameter,key=lambda x:(-x[0],x[1]))
    rank_suspicious_transactions = [line[1] for line in suspicious_transactions_parameter]
    suspicious_transactions_by_line = [transaction for sublist in rank_suspicious_transactions for transaction in sublist]
    return suspicious_transactions_by_line

suspicious_transactions_by_line = sort_suspicious_transactions(suspicious_transactions, score_dict)

print("SUSPICIOUS ID's")
suspicious_transactions_ids = []

for suspicious_transaction in suspicious_transactions_by_line:
    if suspicious_transaction[1] != "sender":
        suspicious_transactions_ids.append(suspicious_transaction[1])

for suspicious_transaction in suspicious_transactions_ids:
    print(suspicious_transaction)

sc.stop()