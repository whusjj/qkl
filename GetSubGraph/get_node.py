from goplus.address import Address
from utils.extract_solidity import *
from web3 import Web3, HTTPProvider
from web3.types import RPCEndpoint

keys_to_check = [
    'blacklist_doubt',
    'blackmail_activities',
    'cybercrime',
    'darkweb_transactions',
    'fake_kyc',
    'financial_crime',
    'honeypot_related_address',
    'malicious_mining_activities',
    'mixer',
    'money_laundering',
    'number_of_malicious_contracts_created',
    'phishing_activities',
    'sanctioned',
    'stealing_attack'
]


def decode_goplus(data):
    # print(type(data))
    result = data.get('result', {})
    if not result:
        return None

    results_list = [key.replace('_', ' ') for key in keys_to_check if result.get(key) == '1']

    # 如果结果列表为空，返回null
    if not results_list:
        return None

    # 否则返回结果列表拼接后的字符串
    return ', '.join(results_list)

def goplus_account(addr):
    goplus_data = Address(access_token=None).address_security(
    address=addr)
    goplus_data = goplus_data.to_dict()
    return str(decode_goplus(goplus_data))


def is_contract_address(address, w3):
    # 使用Web3接口查询地址对应的代码
    code = w3.eth.getCode(address)

    # 如果代码长度大于2（至少有一个字节的代码），则为合约地址，否则为EOA账户
    if len(code) > 2:
        return True
    else:
        return False

def is_contract(addr,w3):
    is_contract = is_contract_address(w3.toChecksumAddress(addr), w3)
    return is_contract

def solidity_assembly(addr, w3):
    runtime_bytecode = w3.eth.get_code(Web3.toChecksumAddress(addr),'latest')
#     cfg = CFG(runtime_bytecode)
    functions = get_functionFromByte(runtime_bytecode)
    return str(functions)

def get_node_contract(addr, w3):
    dict = {}
    dict["is_contract"] = True
    dict["goplus"] = goplus_account(addr)
    dict["code"] = solidity_assembly(addr, w3)
    return dict

def decode_account(addr, w3):
    balance = w3.eth.get_balance(Web3.toChecksumAddress(addr), "latest")
    nonce = w3.eth.get_transaction_count(Web3.toChecksumAddress(addr), "latest")
    return balance, nonce

def get_node_eoa(addr, w3):
    dict = {}
    dict["is_contract"] = False
    dict["goplus"] = goplus_account(addr)
    dict["balance"], dict["nonce"] = decode_account(addr, w3)
    return dict


def get_node(addr, w3):
    if is_contract(addr,w3):
        dict = get_node_contract(addr, w3)
    else:
        dict = get_node_eoa(addr, w3)
    return dict


# w3 = Web3(HTTPProvider('https://docs-demo.quiknode.pro/'))
# print(get_node(w3.toChecksumAddress("0x65aF626611666E0E6B838F8Aa0cE55eb7D7f046e"),w3))


