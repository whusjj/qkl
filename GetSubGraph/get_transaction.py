from web3 import Web3, HTTPProvider
import pandas as pd
from get_node import *

def get_transaction_info1(tx_hash, w3):
    # 查询交易信息
    tx = w3.eth.get_transaction(tx_hash)

    # 去除所有哈希信息
    filtered_tx = {key: value for key, value in tx.items() if not isinstance(value, bytes)}

    return filtered_tx


def get_edge_info(tx_hash, w3):
    result = get_transaction_info1(tx_hash, w3)
    # filtered_dict = {key: value for key, value in result.items() if key not in ["from", "to"]}
    filtered_dict = {key: value for key, value in result.items() if key not in ["input", "from", "to"]}
    return filtered_dict

#
# tx_hash = "0xb1fac2cb5074a4eda8296faebe3b5a3c10b48947dd9a738b2fdf859be0e1fbaf"
# w3 = Web3(HTTPProvider('https://docs-demo.quiknode.pro/'))
# # trx_result = w3.eth.get_transaction(tx_hash)
# # print(trx_result)
#
# print(get_edge_info(tx_hash, w3))
