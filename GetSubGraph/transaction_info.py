from web3 import Web3, HTTPProvider
import pandas as pd
from get_node import *


def byte_string_to_hex(byte_string):
    # 解码字节字符串为十六进制字符串
    hex_string = byte_string.decode('unicode_escape').encode('utf-8').hex()

    # 删除前缀 'b' 和单引号 ''
    hex_string = hex_string.replace('b', '').replace("'", "")

    # 添加 '0x' 前缀并返回结果
    return '0x' + hex_string


def get_transaction_info(tx_hash, w3):
    trx_result = w3.eth.get_transaction(tx_hash)

    from_addr = trx_result["from"]
    to_addr = trx_result["to"]
    return from_addr, to_addr

def get_related_log_event(tx_hash, w3, amount, contract_period, eoa_period):
    related_node = []
    related_node_tx_hash = []
    is_from = []
    trx_result = w3.eth.get_transaction(tx_hash)
    print(trx_result)
    print("****************************")
    from_addr = trx_result["from"]
    to_addr = trx_result["to"]
    block = trx_result["blockNumber"]
    # tx_time = trx_result["tx_time"]
    print("contract collecting...")
    period = contract_period if is_contract(to_addr, w3) else eoa_period
    # print(is_contract(from_addr, w3))
    # print(period)
    filter = {
        'fromBlock': block - period,
        'toBlock': block,
        'address': to_addr,
        # 'topics': ['0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef']
        'topics': []
    }
    logs = w3.eth.get_logs(filter)
    num = -amount
    for log in logs[num:]:
        addr1, addr2 = get_transaction_info(log["transactionHash"], w3)
        if addr1 not in related_node:
            related_node.append(addr1)
            related_node_tx_hash.append(byte_string_to_hex(log["transactionHash"]))
            is_from.append(0)
        if addr2 not in related_node:
            related_node.append(addr2)
            related_node_tx_hash.append(byte_string_to_hex(log["transactionHash"]))
            is_from.append(0)
    print("eoa collecting...")
    period = contract_period if is_contract(from_addr, w3) else eoa_period
    # print(is_contract(to_addr, w3))
    # print(period)
    filter = {
        'fromBlock': block - period,
        'toBlock': block,
        'address': from_addr,
        # 'topics': ['0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef']
        'topics': []
    }
    logs = w3.eth.get_logs(filter)
    num = -amount
    for log in logs[num:]:
        addr1, addr2 = get_transaction_info(log["transactionHash"], w3)
        if addr1 not in related_node:
            related_node.append(addr1)
            related_node_tx_hash.append(byte_string_to_hex(log["transactionHash"]))
            is_from.append(1)
        if addr2 not in related_node:
            related_node.append(addr2)
            related_node_tx_hash.append(byte_string_to_hex(log["transactionHash"]))
            is_from.append(1)
    return related_node, related_node_tx_hash, is_from

def get_related_log_event_entry(tx_hash, w3):
    pace = 4
    contract_pace = 1000
    eoa_pace = 10000
    related_node, related_hash, is_contract = get_related_log_event(tx_hash, w3, pace, contract_pace, eoa_pace)
    df = pd.DataFrame({
        'Related Node': related_node,
        'Related Hash': related_hash,
        'Is From': is_contract
    })
    return df


# tx_hash = "0xb1fac2cb5074a4eda8296faebe3b5a3c10b48947dd9a738b2fdf859be0e1fbaf"
# w3 = Web3(HTTPProvider('https://docs-demo.quiknode.pro/'))
# df = get_related_log_event_entry(tx_hash,w3)
# # # subdf1 = df[df["Is From"] == 1]
# # # print(subdf1)
# # print(df["Related Hash"])
# trx_result = w3.eth.get_transaction(tx_hash)
# print(type(trx_result))
# print(trx_result)
# my_dict = {}
# my_dict["blockNumber"] = trx_result["blockNumber"]
# # dict[]
# my_dict["gas"] = trx_result["gas"]
# my_dict["gasPrice"] = trx_result["gasPrice"]
# my_dict["nonce"] = trx_result["nonce"]
# print(my_dict)

