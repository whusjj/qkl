#  Copyright [2022] [MA WEI @ NTU], ma_wei@ntu.edu.sg

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# from slither.slither import Slither  
import tempfile
import json
import requests
import os
#from web3_input_decoder.utils import hex_to_bytes, get_types_names

# def extract_func(contract_source_file, function_signature):
#     slither = Slither( contract_source_file )  
#     for contract in slither.contracts:  
#         print('Contract: '+ contract.name)  
#         for function in contract.functions:  
#             print('Function: {}'.format(function.signature))  
#             print(function.source_mapping.lines)
#             for e in function.nodes:
#                 print(e.file_scope)
   

def get_url(contract_addr, platform='ethereum', action="sourcecode", api_key="VRP2ARWTUPZYYZX6QJMW14893PZMJ26N4B"):
    actions = {"sourcecode":"getsourcecode", "abi":"getabi"}
    if platform == 'ethereum':
        url = 'https://api.etherscan.io/api'
        api_key= 'VRP2ARWTUPZYYZX6QJMW14893PZMJ26N4B'

    elif platform == 'polygon':
        url = 'https://api.polygonscan.com/api'
        api_key= ''

    elif platform == 'fantom':
        url = 'https://api.ftmscan.com/api'
        api_key= ''

    elif platform == 'bsc':
        url = 'https://api.bscscan.com/api'
        api_key= ''

    elif platform == 'aave':
        url = 'https://api.snowtrace.io/api'
        api_key= ''

    elif platform == 'corno':
        url = 'https://api.cornoscan.com/api'
        api_key= ''

    elif platform == 'aurora':
        url = 'https://api.aurorascan.dev/api'
        api_key= ''

    return f'{url}?module=contract&action={actions[action]}&address={contract_addr}&apikey={api_key}'

def default_ecr721abi():
    return json.load(open("utils/ecr721.abi", "r"))

def default_erc20abi():
    return json.load(open("utils/ecr721.abi", "r"))

def save_to_file(filename, source):
    if '/' in filename:
        idx = filename.rfind('/')
        filepath = filename[:idx]
        os.makedirs(filepath, exist_ok=True)

    with open(filename, 'w+') as f:
        f.write(source)


contract_data="data"
def download_contract_sourcecode(contract_addr, platform='ethereum', action="sourcecode", store=False, api_key="TRXKP7B8KWWZE6YQRHXFH4SGQQ5M9EE5MM"):
    res_source = {}
    res = multiple_trial_get(get_url(contract_addr, platform, action, api_key=api_key))
    res_json = json.loads(res.content)
    contractName = res_json['result'][0]["ContractName"]
    if contractName:
        name = f'{contractName}'
    else:
        name = contract_addr
    source_code = res_json.get('result')
    
    source = {}
    source_str = source_code[0].get('SourceCode')
    if source_str.startswith("{{"):
        source = json.loads(source_str[1:-1]).get('sources')
    elif source_str.startswith("{"):
        source = json.loads(source_str)
    else:
        source[f'{name}.sol'] = {'content': source_str}

    for filename, data in source.items():
        content = data.get('content')
        #if filename.startswith('@'):
        #    continue

        if filename.startswith('/'):
            filename = filename[1:]
        filename = filename.split('/')[-1]
        res_source[filename] = content
    contract_path=os.path.join(contract_data, contract_addr)
    if store:
        os.makedirs( contract_path, exist_ok=True )
        for fname in res_source:
            with open( os.path.join(contract_path, fname), "w" ) as f:
                f.write( res_source[fname] )
    if len(contractName) and f"{contractName}.sol" not in res_source:
        contractName =  list(res_source.items())[0][0][:-4]
    return contractName, res_json, res_source, contract_path

def dowload_contract_proxy_sourcecode(contract_addr, platform='ethereum', action="sourcecode", store=False, api_key='TRXKP7B8KWWZE6YQRHXFH4SGQQ5M9EE5MM'):
    contractName, res_json, source_code, contract_path = download_contract_sourcecode(contract_addr, store=False, api_key=api_key)


    imp_addr = get_implementation_addr(contract_addr)
    imp_contractName, imp_res_json, imp_source_code, imp_contract_path = None, None, {}, None
    if imp_addr != contract_addr:
        imp_contractName, imp_res_json, imp_source_code, imp_contract_path = download_contract_sourcecode(imp_addr, store=False, api_key=api_key)
    
    source_code.update(imp_source_code )
    contractName = imp_contractName if imp_contractName else contractName
    return contractName, [res_json, imp_res_json], source_code, imp_contract_path

#print(tx.logs[0])
def even_sig_url(endpoint):
    return f'https://www.4byte.directory/api/v1/event-signatures/?hex_signature={endpoint}'

#from eth_abi.abi import decode

import time
def decode_log(w3, log):
    # Get ABI of contract
    addr = log["address"]
    # Get event signature of log (first item in topics array)
    receipt_event_signature_hex = w3.toHex(log["topics"][0])
   
    
    query_sig = even_sig_url(receipt_event_signature_hex)
    try:
        response = multiple_trial_get(query_sig)
        #print(response.content)
        #print(response.status_code)
        results = json.loads(response.content)
        results = results['results']
    except:
        time.sleep(5)
        response = multiple_trial_get(query_sig)
        if response.status_code == 200:
            results = json.loads(response.content)
            results = results['results']
        else:
            return ""
        #json.loads(requests.get(query_sig).content)
        #raise ValueError('A very specific bad thing happened.') 
    if len(results) == 0:
        return ""
    text_signature = results[0]["text_signature"]
   
    return text_signature

    

# def get_abi(addr, ETHERSCAN_API_KEY="VRP2ARWTUPZYYZX6QJMW14893PZMJ26N4B"):# Get ABI for smart contract NOTE: Use "to" address as smart contract 'interacted with'
#     abi_endpoint = f"https://api.etherscan.io/api?module=contract&action=getabi&address={addr}&apikey={ETHERSCAN_API_KEY}"
#     #print(abi_endpoint)
#     abi = json.loads(requests.get(abi_endpoint).text)
#     #print(abi)
#     return abi

def multiple_trial_get(sourcecode_endpoint):
    response = requests.get(sourcecode_endpoint)
    trial = 0
    while(trial<5 and response.status_code!=200):
        time.sleep(1)
        response = requests.get(sourcecode_endpoint)
        trial+=1
    return response

def get_implementation_addr(proxy_addr, ETHERSCAN_API_KEY="VRP2ARWTUPZYYZX6QJMW14893PZMJ26N4B"):
    sourcecode_endpoint = f"https://api.etherscan.io/api?module=contract&action=getsourcecode&address={proxy_addr}&apikey={ETHERSCAN_API_KEY}"
    response = multiple_trial_get(sourcecode_endpoint)
    res = json.loads(response.content)
    #h=res.get('result')
    #print(h)
    impl = res.get('result')[0].get('Implementation')
    if impl == proxy_addr:
        return impl
    if impl:
        return get_implementation_addr(impl)
    else:
        return proxy_addr


def get_abi(addr, ETHERSCAN_API_KEY="VRP2ARWTUPZYYZX6QJMW14893PZMJ26N4B"):# Get ABI for smart contract NOTE: Use "to" address as smart contract 'interacted with'
    impl_addr = get_implementation_addr(addr)
    abi_endpoint = f"https://api.etherscan.io/api?module=contract&action=getabi&address={impl_addr}&apikey={ETHERSCAN_API_KEY}"
    response = multiple_trial_get(abi_endpoint)
    return json.loads(response.content)
    #return abi


import sys
from evm_cfg_builder.cfg import CFG, Function
def output_to_dotstring(func) -> None:

    if func.key == Function.DISPATCHER_ID:
        return output_dispatcher_to_dot(func)
        
    lines = []
    #f.write("digraph{\n")
    lines.append("digraph{\n")
    for basic_block in func.basic_blocks:
        instructions_ = [f"{hex(ins.pc)}:{str(ins)}" for ins in basic_block.instructions]
        instructions = "\n".join(instructions_)

        # f.write(f'{basic_block.start.pc}[label="{instructions}"]\n')
        lines.append(f'{basic_block.start.pc}[label="{instructions}"]\n')

        for son in basic_block.outgoing_basic_blocks(func.key):
            # f.write(f"{basic_block.start.pc} -> {son.start.pc}\n")
            lines.append(f"{basic_block.start.pc} -> {son.start.pc}\n")

    # f.write("\n}")
    lines.append("\n}")
    return "".join(lines)


def output_dispatcher_to_dot(func) -> None:
    #with open(f"{base_filename}{self.name}.dot", "w", encoding="utf-8") as f:
    lines = []
    #f.write("digraph{\n")
    lines.append("digraph{\n")
    for basic_block in func.basic_blocks:
        instructions_ = [f"{hex(ins.pc)}:{str(ins)}" for ins in basic_block.instructions]
        instructions = "\n".join(instructions_)

        #f.write(f'{basic_block.start.pc}[label="{instructions}"]\n')
        lines.append(f'{basic_block.start.pc}[label="{instructions}"]\n')
        for son in basic_block.outgoing_basic_blocks(func.key):
            #f.write(f"{basic_block.start.pc} -> {son.start.pc}\n")
            lines.append(f"{basic_block.start.pc} -> {son.start.pc}\n")

        # if not basic_block.outgoing_basic_blocks(func.key):
        #     if basic_block.ends_with_jump_or_jumpi():
        #         logger.error(f"Missing branches {self.name}:{hex(basic_block.end.pc)}")
    for function in func._cfg.functions:
        if function != func:
            #f.write(f'{function.start_addr}[label="Call {function.name}"]\n')
            lines.append(f'{function.start_addr}[label="Call {function.name}"]\n')

    lines.append("\n}")
    return "".join(lines)

def get_functionFromByte(runtime_bytecode):
    cfg = CFG(runtime_bytecode)
    res = {}
    for function in sorted(cfg.functions, key=lambda x: x.start_addr):
        #print(f"Function {function.name}")
        f_dot_str = output_to_dotstring(function)
        # Each function may have a list of attributes
        # An attribute can be:
        # - payable
        # - view
        # - pure
        attributes = " "
        if sorted(function.attributes):
            #print("\tAttributes:")
            for attr in function.attributes:
                #print(f"\t\t-{attr}")
                attributes += f"{attr} "
        function_dec = f"function {function.name} {attributes}; "
        res[function.name] = function_dec
        # print("\n\tBasic Blocks:")
       
    return res