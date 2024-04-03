import numpy as np
from get_random import *
from get_sub import *
import os
import pandas as pd
from for_init1 import *
from get_node import *
from transaction_info import *
from trace import *
from trace_methods import *
from get_transaction import *


def main():
    file = "../data/ethereum9月主交易-5.csv"

    w3 = Web3(HTTPProvider('https://docs-demo.quiknode.pro/'))
    if 'df' not in locals():
        df = pd.read_csv(file)
        # 执行接下来的数据处理或分析操作
    else:
        print("DataFrame 'df' already exists.")
    flag1 = 0
    if "疑似" in file:
        flag1 = 1
    print(flag1)
    transactions = df["tx_hash"]
    result = generate_random_integers(0, len(transactions), 30)
    transactions = get_sublist_by_index(transactions, result)
    print(transactions)
    for tx_hash in transactions:
        # 初始化
        graph = init_graph(flag1)
        node_index, edge_index = init_params()
        # 得到两个节点用来扩展
        from_addr, to_addr = get_transaction_info(tx_hash, w3)
        # 得到calltrace轨迹
        trace = get_edge_trace(tx_hash)
        trace_dict = trace_to_dict(trace)
        trace_info, node_list = get_node_trace_info(trace_dict, trace)
        new_text = "\n".join([line.split("params")[0] for line in trace_info.split("\n")])
        graph.global_context = new_text
        #         print(trace_info)
        # 得到from节点的相关信息
        dict_tmp = get_node(w3.toChecksumAddress(from_addr), w3)
        #         dict_tmp["calltrace"] = replace_tabs_and_newlines(trace_info)
        target_from_node = [dict_tmp]
        # 添加节点
        graph.add_node(target_from_node, node_index)
        node_index += 1  # 处理node_index
        # 得到from节点的相关信息
        target_to_node = [get_node(w3.toChecksumAddress(to_addr), w3)]
        # 添加节点
        graph.add_node(target_to_node, node_index)
        node_index += 1  # 处理node_index

        # 添加边：
        new_edge_indices = [[0, 1]]
        edge_dict = get_edge_info(tx_hash, w3)
        new_edge_attributes = [edge_dict]
        graph.add_edge(new_edge_indices, new_edge_attributes)
        edge_index += 1

        for address in node_list[2:]:
            related_to = [get_node(w3.toChecksumAddress(address), w3)]
            graph.add_node(related_to, node_index)
            new_edge_indices = [[1, node_index]]
            new_edge_attributes = ["calltrace"]
            graph.add_edge(new_edge_indices, new_edge_attributes)
            edge_index += 1
            node_index += 1
        # 通过 log event进行扩展
        data_df = get_related_log_event_entry(tx_hash, w3)
        from_df = data_df[data_df["Is From"] == 1]
        to_df = data_df[data_df["Is From"] == 0]
        print(data_df)
        for address, txhash in zip(from_df["Related Node"], from_df["Related Hash"]):
            related_from = [get_node(w3.toChecksumAddress(address), w3)]
            graph.add_node(related_from, node_index)  # 处理node_index
            new_edge_indices = [[0, node_index]]
            edge_dict = get_edge_info(tx_hash, w3)
            new_edge_attributes = [edge_dict]
            graph.add_edge(new_edge_indices, new_edge_attributes)
            edge_index += 1
            node_index += 1

        for address, txhash in zip(to_df["Related Node"], to_df["Related Hash"]):
            related_to = [get_node(w3.toChecksumAddress(address), w3)]
            graph.add_node(related_to, node_index)
            new_edge_indices = [[1, node_index]]
            edge_dict = get_edge_info(tx_hash, w3)
            new_edge_attributes = [edge_dict]
            graph.add_edge(new_edge_indices, new_edge_attributes)
            edge_index += 1
            node_index += 1
        # 呈现一个交易的结果
        graph.print_pyg()
        save_path = f"result/{tx_hash}.pt"
        torch.save(graph.get_pyg(), save_path)
        print(f"PyG 数据集已保存")

if __name__ == "__main__":
    main()