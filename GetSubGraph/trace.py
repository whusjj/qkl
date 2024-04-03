# -*- coding:utf-8 -*-
import math

import numpy as np
import pandas as pd
import enum
import json
from typing import List, Dict

from web3 import Web3, HTTPProvider
from web3.types import RPCEndpoint


class CallType(enum.Enum):
    CALL = "CALL"
    STATIC_CALL = "STATICCALL"
    DELEGATE_CALL = "DELEGATECALL"

    SELF_DESTRUCT = "SELFDESTRUCT"
    CREATE = "CREATE"
    CREATE2 = "CREATE2"
    CALLCODE = "CALLCODE"

    UNKNOWN = "UNKNOWN"

    @staticmethod
    def get_by_key(key):
        if key == 'CALL':
            return CallType.CALL
        elif key == 'STATICCALL':
            return CallType.STATIC_CALL
        elif key == 'DELEGATECALL':
            return CallType.DELEGATE_CALL
        elif key == "SELFDESTRUCT":
            return CallType.SELF_DESTRUCT
        elif key == "CREATE":
            return CallType.CREATE
        elif key == "CREATE2":
            return CallType.CREATE2
        elif key == "CALLCODE":
            return CallType.CALLCODE
        else:
            return CallType.UNKNOWN


class Call:
    idx: int
    type: CallType
    parent: object
    from_address: str
    to_address: str
    value: int
    input: str
    output: str
    gas: int
    gas_used: int
    time_takes: str
    sub_calls: List
    json_data: Dict
    # 闪电贷信息
    flash_loan_data: Dict

    def set_type(self, call_type: str):
        self.type = CallType.get_by_key(call_type)

    @property
    def json_str(self) -> str:
        return json.dumps(self.json_data)

    @staticmethod
    def load_from_json(json_data, parent=None, idx=0):
        if not json_data:
            return None
        call = Call()
        if not parent:
            call.idx = str(idx)
        else:
            call.idx = f"{parent.idx}_{idx}"
        call.parent = parent
        call.json_data = json_data
        call.set_type(json_data.get('type'))
        call.from_address = json_data.get('from')
        call.to_address = json_data.get('to')
        call.value = int(json_data.get('value', '0x0'), 16)
        call.input = json_data.get('input')
        call.output = json_data.get('output')
        call.gas = int(json_data.get('gas', '0x0'), 16)
        call.gas_used = int(json_data.get('gasUsed', '0x0'), 16)
        call.time_takes = json_data.get('time')
        call.flash_loan_data = None
        if len(json_data.get('calls', [])) >= 0:
            sub_calls = []
            for sub_idx, sub_call in enumerate(json_data.get('calls', [])):
                sc = Call.load_from_json(sub_call, parent=call, idx=sub_idx)
                sub_calls.append(sc)
            call.sub_calls = sub_calls
        return call
    
def read_entire_file(filename):
    content = ""
    file = open(filename, 'r')
    content = file.read()
    file.close()
    return content

def get_web3(chain_symbol):
    if chain_symbol == "eth":
        return Web3(HTTPProvider("https://docs-demo.quiknode.pro/"))
    else:
        raise ValueError(f"unsupported chain: {chain_symbol}")


def get_calltrace(web3: Web3, tx: str):
    trace_res = web3.provider.make_request(RPCEndpoint('debug_traceTransaction'), [tx, {"tracer": "callTracer"}])
    return Call.load_from_json(trace_res.get("result"))


class FunctionCall:
    name: str
    signature: str
    parameters: str
    returns: str

    def __init__(self, name, parameters, returns, signature=None):
        self.name = name
        self.parameters = parameters
        self.returns = returns
        if not signature:
            self.signature = Web3.keccak(text=f"{name}({parameters})").hex()[:10]
        else:
            self.signature = signature

    def __str__(self):
        return f"function {self.name}({self.parameters}) returns ({self.returns}) signature:{self.signature}"


def parse_abi(abi_str: str) -> List[FunctionCall]:
    abi = json.loads(abi_str)
    functions = []
    for obj in abi:
        if obj.get('type') == "function":
            name = obj.get("name")
            parameters = ",".join([x.get("type") for x in obj.get("inputs")])
            returns = ",".join([x.get("type") for x in obj.get("outputs")])
            functions.append(FunctionCall(name, parameters, returns))
    return functions


def parse_value(type, value):
    if type.startswith("address") or type.startswith("uint") or type.startswith("bool"):
        if isinstance(value, tuple):
            return f'[{",".join([str(x) for x in list(value)])}]'
        return str(value)
    if type.startswith("bytes"):
        if isinstance(value, tuple):
            return ["bytes"]
        return "bytes"
    return str(value)


def decode_parameter(web3, parameter_def, inputs):
    if parameter_def:
        try:
            param_defs = parameter_def.split(",")
            params = web3.codec.decode(param_defs, bytes.fromhex(inputs))
            format_values = []
            for t, v in dict(zip(param_defs, list(params))).items():
                format_values.append(parse_value(t, v))
            return format_values
        except Exception as e:
            raise e
    return ""


def decode_returns(web3, returns_def, output):
    if returns_def:
        try:
            returns_def = returns_def.split(",")
            returns_value = web3.codec.decode(returns_def, bytes.fromhex(output))
            format_values = []
            for k, v in dict(zip(returns_def, list(returns_value))).items():
                format_values.append(parse_value(k, v))
            return format_values
        except:
            return output
    return ""


def get_valid_signatures():
    df = pd.read_csv("function_signatures.csv")
    df = df.where(pd.notnull(df), None)
    functions = []
    for _, row in df.iterrows():
        functions.append(FunctionCall(name=row[1].split('(')[0], parameters=row[2], returns=row[3], signature=row[0]))
    return functions


def pretty_print(web3: Web3, calltrace: Call, functions=None):
    functions = get_valid_signatures()
    valid_function_sigs = dict(zip([x.signature for x in functions], functions))
    edge_trace = []

    def get_address_symbol(addr):
        return addr

    def write_call(call, prefix="", edge_trace=[], file=None):
        _from = get_address_symbol(call.from_address)
        _to = get_address_symbol(call.to_address)
        if call.input == "0x":
            file.write(f"{prefix}from: {_from}\tto: {_to}\tcall: send ether({round(call.value / math.pow(10, 18), 6)})\n")
        else:
            func_sig = call.input[:10]
            if func_sig in valid_function_sigs:
                func = valid_function_sigs.get(func_sig)
                param_values = decode_parameter(web3, func.parameters, call.input[10:])
                if call.output:
                    return_values = decode_returns(web3, func.returns, call.output[2:])
                else:
                    return_values = ""
                param_values_str = [str(param) for param in param_values]  # 将参数值转换为字符串列表
                file.write(f"{prefix}from: {_from}\tto: {_to}\t"
                           f"call: {func.name}\tparams:{','.join(param_values_str)}\t"  # 使用参数值的字符串列表
                           f"returns: {','.join(return_values)}\n")
                # file.write(f"{prefix}from: {_from}\tto: {_to}\t"
                #            f"call: {str(func.name)}\tparams:{','.join(param_values)}\t"
                #            f"returns: {','.join(return_values)}\n")
            else:
                file.write(f"{prefix}from: {_from}\tto: {_to}\tcall: {call.input}\n")
            for sub_call in call.sub_calls:
                write_call(sub_call, prefix + "-", file=file)

    with open('trx_keep.txt', 'w') as file:
        write_call(calltrace, file=file)


def get_edge_trace(trx_hash):
    w3 = get_web3("eth")
    calltrace = get_calltrace(w3, trx_hash)
    if calltrace:
        pretty_print(w3, calltrace)
        file_content = read_entire_file('trx_keep.txt')
        return file_content
    else:
        print("none calltrace")

def replace_tabs_and_newlines(input_string):
    replaced_string = input_string.replace('\t', ' ').replace('\n', ' ')
    return replaced_string